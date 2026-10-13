// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Full HTTP routing path backed by a real in-memory Indexer gRPC server.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use sgl_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sgl_kv_indexer::pb::kv_indexer_server::{KvIndexer, KvIndexerServer};
use sgl_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvAction,
    ExternalKvActionType, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse,
    MatchExternalKvPrefixRequest, MatchExternalKvPrefixResponse, MatchExternalKvRequest,
    MatchExternalKvResponse, TierType,
};
use sgl_kv_indexer::{
    server_builder, GrpcPrefixIndex, InMemoryKvIndexerBackend, KvIndexerService, PrefixIndex,
    PrefixIndexConfig,
};
use sgl_router::config::{AffinityConfig, CachePrefixProvider, PolicyKind};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::request_tokens_for;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::state::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use tokio_stream::wrappers::TcpListenerStream;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, MODEL};
use crate::common::mock_worker::MockWorker;

#[tokio::test]
async fn external_indexer_routes_to_the_cached_worker() {
    let cached = MockWorker::start(vec![]).await;
    let uncached = MockWorker::start(vec![]).await;
    let mut cfg = config();
    cfg.model.policy = PolicyKind::CacheAware;
    cfg.model
        .cache_aware
        .as_mut()
        .expect("fixture includes cache-aware configuration")
        .prefix_provider = CachePrefixProvider::Indexer;
    cfg.model.affinity = Some(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(0),
        cache_candidate_min_workers: 1,
        cache_candidate_ratio: 1.0,
        cache_candidate_max_workers: 1,
        ..Default::default()
    });
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello there friend"}],
    });
    let tokens = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body)
        .expect("test prompt tokenizes");
    let hashes = compute_block_hashes(&tokens.ids, 1);
    assert!(!hashes.is_empty());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        server_builder()
            .add_service(KvIndexerService::new(InMemoryKvIndexerBackend::new()).into_server())
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    let mut indexer = KvIndexerClient::connect(endpoint.clone()).await.unwrap();
    indexer
        .apply_external_kv_batch(ApplyExternalKvBatchRequest {
            worker_id: "cached-worker".into(),
            seq: 1,
            actions: vec![ExternalKvAction {
                r#type: ExternalKvActionType::ActionReport as i32,
                tier: TierType::TierHbm as i32,
                hashes: hashes.clone(),
                component_masks: Vec::new(),
                block_sizes: Vec::new(),
                parent_block_hash: None,
            }],
            worker_address: cached.url.clone(),
            cache_spec: None,
        })
        .await
        .unwrap();

    let registry = Arc::new(WorkerRegistry::default());
    for url in [&cached.url, &uncached.url] {
        registry
            .add(WorkerSpec {
                id: WorkerId(url.clone()),
                url: url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId(MODEL.into())],
                ..Default::default()
            })
            .unwrap();
    }
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    let policies =
        Arc::new(build_registry(&cfg, Arc::new(HashTree::new()), Arc::clone(&oracle)).unwrap());
    let mut ctx = AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    );
    ctx.prefix_index = Some(Arc::new(
        GrpcPrefixIndex::new(PrefixIndexConfig {
            endpoint,
            query_deadline: Duration::from_secs(1),
            max_inflight: 4,
            p2p_candidate_hosts: 0,
        })
        .unwrap(),
    ));
    ctx.block_size_oracle = oracle;

    let app = build_router(Arc::new(ctx));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(cached.captured.lock().unwrap().last_body.is_some());
    assert!(uncached.captured.lock().unwrap().last_body.is_none());

    server.abort();
}

/// Records what the Router's client actually sent, so the flag → wire seam has a
/// test of its own. Everything else about this hop can be right while
/// `p2p_candidate_hosts` is silently dropped, and a P2P deployment would then
/// look merely "not helpful" instead of broken.
#[derive(Clone, Default)]
struct RecordingIndexer {
    seen: Arc<std::sync::Mutex<Vec<u32>>>,
}

#[tonic::async_trait]
impl KvIndexer for RecordingIndexer {
    async fn match_external_kv_prefix(
        &self,
        request: tonic::Request<MatchExternalKvPrefixRequest>,
    ) -> Result<tonic::Response<MatchExternalKvPrefixResponse>, tonic::Status> {
        self.seen
            .lock()
            .expect("request recorder")
            .push(request.into_inner().p2p_candidate_hosts);
        Ok(tonic::Response::new(Default::default()))
    }

    async fn apply_external_kv_batch(
        &self,
        _request: tonic::Request<ApplyExternalKvBatchRequest>,
    ) -> Result<tonic::Response<ApplyExternalKvBatchResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used by this test"))
    }

    async fn match_external_kv(
        &self,
        _request: tonic::Request<MatchExternalKvRequest>,
    ) -> Result<tonic::Response<MatchExternalKvResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used by this test"))
    }

    async fn get_external_kv_hit_counts(
        &self,
        _request: tonic::Request<GetExternalKvHitCountsRequest>,
    ) -> Result<tonic::Response<GetExternalKvHitCountsResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("not used by this test"))
    }
}

/// The configured P2P candidate count must reach the Indexer: `main.rs`'s
/// `prefix_index_config` turns it into `PrefixIndexConfig`, and this asserts the
/// last hop, from that config to the query the Indexer answers. Both values are
/// checked, because "0" is the deployment that must not pay for the dimension.
#[tokio::test]
async fn router_client_sends_the_configured_p2p_candidate_count() {
    let recorder = RecordingIndexer::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn({
        let recorder = recorder.clone();
        async move {
            server_builder()
                .add_service(KvIndexerServer::new(recorder))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap();
        }
    });

    let indexer = |p2p_candidate_hosts: u32| {
        GrpcPrefixIndex::new(PrefixIndexConfig {
            endpoint: endpoint.clone(),
            query_deadline: Duration::from_secs(2),
            max_inflight: 4,
            p2p_candidate_hosts,
        })
        .expect("test endpoint is valid")
    };
    indexer(3)
        .match_prefix(vec![1, 2, 3])
        .await
        .expect("query reaches the indexer");
    indexer(0)
        .match_prefix(vec![1, 2, 3])
        .await
        .expect("query reaches the indexer");

    assert_eq!(
        recorder.seen.lock().expect("request recorder").clone(),
        vec![3, 0]
    );
    server.abort();
}

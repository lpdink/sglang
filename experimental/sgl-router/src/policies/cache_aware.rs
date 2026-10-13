// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Builds bounded cache-aware candidates from ingress Indexer results.

use crate::config::AffinityConfig;
use crate::policies::admission::CandidateLoads;
use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
use crate::policies::{
    CacheCandidate, CacheCandidateProposal, Policy, PrefillProposal, ProposalKind,
    SelectionContext, SelectionProposal,
};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: AffinityConfig,
}

impl CacheAwarePolicy {
    pub fn new(config: AffinityConfig) -> Self {
        Self { config }
    }

    fn cache_candidate_proposal(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<CacheCandidateProposal> {
        let input_tokens = ctx.input_tokens()?;
        let signal = ctx.external_prefix()?;
        let sgl_kv_indexer::PrefixOutcome::Matched { matches, .. } = &signal.outcome else {
            return None;
        };
        if signal.query_blocks == 0 || workers.is_empty() {
            return None;
        }

        // The #33370 indexer contract routes on the worker address (matched
        // byte-for-byte against registered worker URLs); worker_id is for
        // logs only.
        let by_url: HashMap<&str, &Arc<Worker>> = workers
            .iter()
            .map(|worker| (worker.url.as_str(), worker))
            .collect();
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        for entry in matches {
            let Some(worker) = by_url.get(entry.address.as_str()) else {
                continue;
            };
            // A peer-served extension counts as a serveable prefix: the worker
            // can prefill the request from it. The max is defensive — the
            // contract already makes the P2P prefix an extension of the local
            // one — and with P2P off both fields are 0, so the prefix and every
            // quantity derived below it are exactly the pre-P2P ones.
            let matched_prefix_blocks = cap_matched_prefix_blocks(
                signal.query_blocks,
                entry.matched_prefix_blocks.max(entry.p2p_prefix_blocks),
            );
            if matched_prefix_blocks == 0 || !seen.insert(worker.id.clone()) {
                continue;
            }
            let matched_prefix_tokens = estimate_matched_prefix_tokens(
                input_tokens,
                signal.query_blocks,
                matched_prefix_blocks,
            );
            if !self.passes_cache_gate(input_tokens, matched_prefix_tokens) {
                continue;
            }
            candidates.push(CacheCandidate {
                worker: Arc::clone(worker),
                matched_prefix_tokens,
                uncached_tokens: input_tokens.saturating_sub(matched_prefix_tokens),
                matched_prefix_blocks,
                // Capped at the prefix it serves, so a malformed answer cannot
                // rank a candidate on a transfer it cannot need.
                p2p_fetch_blocks: entry.p2p_fetch_blocks.min(matched_prefix_blocks),
                candidate_range_id: ctx.candidate_range_id().to_string(),
                max_pending_prefill_tokens: None,
            });
        }

        if let Some((selector, request)) = ctx.prefill_cache_bucket() {
            candidates = candidates
                .into_iter()
                .filter_map(|candidate| {
                    selector.prepare_prefill_cache_candidate(candidate, request)
                })
                .collect();
        }

        let limit = self.candidate_limit(workers.len());
        if limit == 0 {
            return None;
        }
        let loads = CandidateLoads::new(
            ctx.load_snapshot(),
            candidates.iter().map(|candidate| &candidate.worker),
        );
        if candidates.len() > limit {
            candidates.select_nth_unstable_by(limit, |left, right| {
                compare_candidate_seed(left, right, &loads)
            });
            candidates.truncate(limit);
        }
        candidates.sort_by(|left, right| compare_candidate_seed(left, right, &loads));
        if candidates.is_empty() {
            return None;
        }
        Some(CacheCandidateProposal {
            candidates,
            cache_switch_margin_tokens: self.config.cache_switch_margin_tokens,
            enable_pressure_guard: self.config.pressure_guard,
            pressure_abs_threshold_tokens: self.config.pressure_abs_threshold_tokens,
            pressure_abs_threshold_ms: self.config.pressure_abs_threshold_ms,
            pressure_rel_threshold: self.config.pressure_rel_threshold,
            worker_queue_limit: self.config.worker_queue_limit,
            saturation_queue_floor: self.config.saturation_queue_floor,
        })
    }

    fn passes_cache_gate(&self, input_tokens: u64, matched_prefix_tokens: u64) -> bool {
        self.config
            .cache_affinity_min_matched_tokens
            .is_none_or(|minimum| matched_prefix_tokens >= minimum)
            && self
                .config
                .cache_affinity_min_match_ratio
                .is_none_or(|minimum| {
                    input_tokens > 0
                        && matched_prefix_tokens as f64 / input_tokens as f64 >= minimum
                })
    }

    fn candidate_limit(&self, worker_count: usize) -> usize {
        let proportional = (self.config.cache_candidate_ratio.clamp(0.0, 1.0) * worker_count as f64)
            .ceil() as usize;
        worker_count
            .min(self.config.cache_candidate_max_workers)
            .min(self.config.cache_candidate_min_workers.max(proportional))
    }
}

fn compare_candidate_seed(
    left: &CacheCandidate,
    right: &CacheCandidate,
    loads: &CandidateLoads<'_>,
) -> Ordering {
    right
        .matched_prefix_tokens
        .cmp(&left.matched_prefix_tokens)
        // A peer-served extension is not free: at an equal serveable prefix the
        // candidate that fetches fewer blocks wins, before load is consulted.
        // With P2P off both sides are 0 and the key drops out.
        .then_with(|| left.p2p_fetch_blocks.cmp(&right.p2p_fetch_blocks))
        .then_with(|| loads.compare_prefill_engines(&left.worker, &right.worker))
        .then_with(|| left.worker.id.0.cmp(&right.worker.id.0))
}

impl Policy for CacheAwarePolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        self.propose(workers, ctx).map(|proposal| proposal.primary)
    }

    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match self.propose_prefill(workers, ctx)? {
            PrefillProposal::Pair(proposal) => Some(proposal),
            PrefillProposal::CacheCandidates(proposal) => {
                let candidate = proposal.candidates.into_iter().next()?;
                Some(
                    SelectionProposal::primary(candidate.worker)
                        .with_kind(ProposalKind::CacheAffinity),
                )
            }
        }
    }

    fn propose_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        if ctx.affinity_lookup_enabled() {
            if let Some(proposal) = self.cache_candidate_proposal(workers, ctx) {
                return Some(PrefillProposal::CacheCandidates(proposal));
            }
        }
        PowerOfTwoChoicesPolicy::new()
            .with_load_control(self.config.min_load_choices, self.config.worker_queue_limit)
            .propose(workers, ctx)
            .map(PrefillProposal::Pair)
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }
}

/// Caps an indexer-supplied matched-block count at the blocks the query
/// actually asked about: a query cannot match more blocks than it contains.
/// Both the token estimate and the diverted-overlap histogram read the capped
/// value, so the clamp lives here rather than at each use.
fn cap_matched_prefix_blocks(query_blocks: usize, matched_prefix_blocks: u32) -> u32 {
    matched_prefix_blocks.min(u32::try_from(query_blocks).unwrap_or(u32::MAX))
}

fn estimate_matched_prefix_tokens(
    input_tokens: u64,
    query_blocks: usize,
    matched_prefix_blocks: u32,
) -> u64 {
    let matched_prefix_blocks = u64::from(cap_matched_prefix_blocks(
        query_blocks,
        matched_prefix_blocks,
    ));
    let query_blocks = u64::try_from(query_blocks).unwrap_or(u64::MAX).max(1);
    input_tokens.saturating_mul(matched_prefix_blocks) / query_blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::state::kv_events::PrefixLookupResult;

    #[test]
    fn matched_token_estimate_caps_untrusted_block_count() {
        assert_eq!(estimate_matched_prefix_tokens(80, 8, 99), 80);
    }

    #[test]
    fn matched_block_cap_is_shared_by_the_estimate_and_the_candidate() {
        // One clamp, two readers: the histogram must never see a block count
        // the token estimate would have thrown away.
        assert_eq!(cap_matched_prefix_blocks(8, 99), 8);
        assert_eq!(cap_matched_prefix_blocks(8, 3), 3);
        assert_eq!(cap_matched_prefix_blocks(0, 3), 0);
    }

    // --- P2P dimension ---

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".into())],
            ..Default::default()
        }))
    }

    /// One indexer answer: how much the worker holds itself, and how far a peer
    /// could extend it.
    struct Hit<'a> {
        worker: &'a Arc<Worker>,
        matched_blocks: u32,
        p2p_blocks: u32,
        p2p_fetch_blocks: u32,
    }

    fn hit(worker: &Arc<Worker>, matched_blocks: u32) -> Hit<'_> {
        Hit {
            worker,
            matched_blocks,
            p2p_blocks: 0,
            p2p_fetch_blocks: 0,
        }
    }

    fn p2p_hit(worker: &Arc<Worker>, matched_blocks: u32, p2p_blocks: u32, fetch: u32) -> Hit<'_> {
        Hit {
            worker,
            matched_blocks,
            p2p_blocks,
            p2p_fetch_blocks: fetch,
        }
    }

    fn prefix_signal(hits: &[Hit<'_>], query_blocks: usize) -> PrefixLookupResult {
        PrefixLookupResult {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: hits
                    .iter()
                    .map(|hit| sgl_kv_indexer::PrefixMatch {
                        address: hit.worker.url.clone(),
                        matched_prefix_blocks: hit.matched_blocks,
                        p2p_prefix_blocks: hit.p2p_blocks,
                        p2p_fetch_blocks: hit.p2p_fetch_blocks,
                        worker_id: hit.worker.id.0.clone(),
                    })
                    .collect(),
                best_prefix_blocks: hits
                    .iter()
                    .map(|hit| hit.matched_blocks.max(hit.p2p_blocks))
                    .max()
                    .unwrap_or(0),
            },
            query_blocks,
            block_hashes: None,
        }
    }

    /// Runs one Cache-Aware selection and returns the candidates it proposed, in
    /// order, with their serveable-prefix bookkeeping.
    fn select(
        policy: &CacheAwarePolicy,
        workers: &[Arc<Worker>],
        signal: &PrefixLookupResult,
        input_tokens: u64,
    ) -> Vec<(String, u64, u32, u32)> {
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(input_tokens)
            .with_external_prefix(Some(signal));
        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(workers, &ctx)
            .expect("a cache signal with matches must propose candidates")
        else {
            panic!("an indexer hit must not fall through to the load pair");
        };
        proposal
            .candidates
            .into_iter()
            .map(|candidate| {
                (
                    candidate.worker.id.0.clone(),
                    candidate.matched_prefix_tokens,
                    candidate.matched_prefix_blocks,
                    candidate.p2p_fetch_blocks,
                )
            })
            .collect()
    }

    /// A policy whose gate accepts any nonzero match, so these tests isolate the
    /// ordering rather than the threshold.
    fn ungated_policy() -> CacheAwarePolicy {
        CacheAwarePolicy::new(AffinityConfig {
            cache_affinity_min_matched_tokens: None,
            ..Default::default()
        })
    }

    #[test]
    fn a_peer_served_prefix_outranks_a_longer_local_one() {
        let local = worker("local");
        let extended = worker("extended");
        let workers = vec![Arc::clone(&local), Arc::clone(&extended)];
        // `local` holds more itself; `extended` reaches further with a peer's
        // help, which is the whole point of the P2P dimension.
        let signal = prefix_signal(&[hit(&local, 7), p2p_hit(&extended, 3, 9, 6)], 10);

        let candidates = select(&ungated_policy(), &workers, &signal, 1_000);

        assert_eq!(
            candidates,
            vec![
                // 900 tokens of the 1_000-token prompt served, 6 blocks pulled.
                ("extended".to_string(), 900, 9, 6),
                ("local".to_string(), 700, 7, 0),
            ]
        );
    }

    #[test]
    fn equal_servable_prefix_prefers_the_cheaper_transfer() {
        let pricey = worker("pricey");
        let cheap = worker("cheap");
        let workers = vec![Arc::clone(&pricey), Arc::clone(&cheap)];
        let signal = prefix_signal(&[p2p_hit(&pricey, 4, 8, 6), p2p_hit(&cheap, 5, 8, 1)], 10);

        let candidates = select(&ungated_policy(), &workers, &signal, 1_000);

        // Same serveable prefix, so the block count the worker must fetch
        // decides — not the load, and not the id.
        assert_eq!(
            candidates,
            vec![
                ("cheap".to_string(), 800, 8, 1),
                ("pricey".to_string(), 800, 8, 6),
            ]
        );
    }

    #[test]
    fn zeroed_p2p_columns_restore_the_local_ranking() {
        let local = worker("local");
        let extended = worker("extended");
        let workers = vec![Arc::clone(&local), Arc::clone(&extended)];
        // The same local prefixes, with the P2P extension reported...
        let with_p2p = prefix_signal(&[hit(&local, 7), p2p_hit(&extended, 3, 9, 6)], 10);
        let proposal_with_p2p = select(&ungated_policy(), &workers, &with_p2p, 1_000);
        assert_eq!(proposal_with_p2p[0].0, "extended");
        // ...and zeroed, which is what every P2P-off client sends (and what
        // backends without a P2P source report). The ranking is the pre-P2P one.
        let zeroed = prefix_signal(&[hit(&local, 7), hit(&extended, 3)], 10);
        let proposal_zeroed = select(&ungated_policy(), &workers, &zeroed, 1_000);
        assert_eq!(
            proposal_zeroed,
            vec![
                ("local".to_string(), 700, 7, 0),
                ("extended".to_string(), 300, 3, 0),
            ]
        );
    }

    #[test]
    fn the_cache_gate_reads_the_servable_prefix() {
        let extended = worker("extended");
        let shallow = worker("shallow");
        let workers = vec![Arc::clone(&extended), Arc::clone(&shallow)];
        let signal = prefix_signal(&[p2p_hit(&extended, 2, 9, 7), hit(&shallow, 4)], 10);
        let policy = CacheAwarePolicy::new(AffinityConfig {
            // Fits the extended prefix (900 of 1_000 tokens) and neither local
            // prefix, so a gate that read the local count would drop both.
            cache_affinity_min_matched_tokens: Some(700),
            ..Default::default()
        });

        let candidates = select(&policy, &workers, &signal, 1_000);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, "extended");
    }

    #[test]
    fn a_p2p_over_report_is_capped_at_the_query_length() {
        let worker = worker("over");
        let workers = vec![Arc::clone(&worker)];
        let signal = prefix_signal(&[p2p_hit(&worker, 2, 99, 99)], 10);

        let candidates = select(&ungated_policy(), &workers, &signal, 1_000);

        // Neither the served prefix nor the transfer it costs can exceed the
        // blocks the query asked about.
        assert_eq!(candidates, vec![("over".to_string(), 1_000, 10, 10)]);
    }
}

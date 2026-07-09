# plans Contract

## Purpose

- Own implementation plans for accepted work that spans multiple files, subsystems, or verification stages.

## Ownership

- Plan documents own proposed sequencing, affected contracts, risks, and verification before implementation.

## Local Contracts

- Plans must derive from the current `SPEC.md`, applicable `AGENTS.md` chain, and source state.
- A plan may resolve implementation sequencing but may not silently settle `SPEC.md` open questions or bypass its banned list.

## Work Guidance

- Identify subsystem ownership, data flow, public interfaces, downstream consumers, and the DOX updates implied by the change.

## Verification

- Before implementation, re-check plan assumptions against current source and config because plans can become stale.

## Child DOX Index

- No child `AGENTS.md` files currently exist.

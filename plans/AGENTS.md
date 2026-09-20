# plans Contract

## Purpose

- Own implementation plans for accepted work that spans multiple files, subsystems, or verification stages.

## Ownership

- Plan documents own proposed sequencing, affected contracts, risks, and verification before implementation.

## Local Contracts

- Plans must derive from the applicable `AGENTS.md` contract chain and current source state.
- A plan may resolve implementation sequencing but may not silently settle an architecture question the owner has not decided, or cross a boundary the contract chain sets — notably the framework split between the two arms.

## Work Guidance

- Identify subsystem ownership, data flow, public interfaces, downstream consumers, and the DOX updates implied by the change.

## Verification

- Before implementation, re-check plan assumptions against current source and config because plans can become stale.

## Child DOX Index

- No child `AGENTS.md` files currently exist.

"""Reserved special tokens for the shared vocabulary."""

MASK_TOKEN = "[MASK]"
PAD_TOKEN = "[PAD]"
END_TOKEN = "[END]"
DELIMITER_TOKEN = "[DELIMITER]"
WIN_TOKEN = "[WIN]"
LOSS_TOKEN = "[LOSS]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

MASK_ID = 0
PAD_ID = 1
END_ID = 2
DELIMITER_ID = 3
WIN_ID = 4
LOSS_ID = 5
BOS_ID = 6
EOS_ID = 7

SPECIAL_TOKENS: dict[str, int] = {
    MASK_TOKEN: MASK_ID,
    PAD_TOKEN: PAD_ID,
    END_TOKEN: END_ID,
    DELIMITER_TOKEN: DELIMITER_ID,
    WIN_TOKEN: WIN_ID,
    LOSS_TOKEN: LOSS_ID,
    BOS_TOKEN: BOS_ID,
    EOS_TOKEN: EOS_ID,
}

SPECIAL_TOKEN_IDS = frozenset(SPECIAL_TOKENS.values())
if SPECIAL_TOKEN_IDS != frozenset(range(len(SPECIAL_TOKENS))):
    raise RuntimeError("special token IDs must be contiguous from zero")

# Content IDs immediately follow the contiguous special-token block. Keeping
# this derived from the registry prevents unnamed holes from entering the
# embedding, output head, corruption prior, or sampler state space again.
CONTENT_TOKEN_OFFSET = len(SPECIAL_TOKENS)

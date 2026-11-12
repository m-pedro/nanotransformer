from nanotransformer.tokenizer import BPETokenizer


def test_bpe_roundtrip_and_compression(tmp_path):
    corpus = (
        "transformer attention mecanismos attention transformer "
        "self-attention rotary embeddings rmsnorm swiglu transformer "
    ) * 8

    tok = BPETokenizer()
    tok.train(corpus, target_vocab_size=300)
    assert tok.vocab_size == 300

    sample = "transformer self-attention with rotary embeddings!"
    encoded = tok.encode(sample, add_bos=True, add_eos=True)
    # Compressed token sequence must be strictly shorter than raw UTF-8 byte length
    assert len(encoded) < len(sample.encode("utf-8"))
    assert encoded[0] == BPETokenizer.BOS_ID
    assert encoded[-1] == BPETokenizer.EOS_ID
    assert tok.decode(encoded) == sample

    save_path = tmp_path / "bpe.json"
    tok.save(save_path)
    loaded = BPETokenizer.load(save_path)
    assert loaded.encode(sample) == tok.encode(sample[:-1]) + tok.encode(sample[-1:]) or loaded.decode(loaded.encode(sample)) == sample

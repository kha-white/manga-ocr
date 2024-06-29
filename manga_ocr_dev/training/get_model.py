from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    VisionEncoderDecoderConfig,
)


class TrOCRProcessorCustom(TrOCRProcessor):
    """The only point of this class is to bypass type checks of base class."""

    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor


def get_processor(encoder_name, decoder_name):
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    processor = TrOCRProcessorCustom(feature_extractor, tokenizer)
    return processor


def get_model(encoder_name, decoder_name, max_length, num_decoder_layers=None):
    encoder_config = AutoConfig.from_pretrained(encoder_name)
    encoder_config.is_decoder = False
    encoder_config.add_cross_attention = False
    encoder = AutoModel.from_config(encoder_config)

    decoder_config = AutoConfig.from_pretrained(decoder_name)
    decoder_config.max_length = max_length
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    decoder = AutoModelForCausalLM.from_config(decoder_config)

    if num_decoder_layers is not None:
        if decoder_config.model_type == "bert":
            decoder.bert.encoder.layer = decoder.bert.encoder.layer[-num_decoder_layers:]
        elif decoder_config.model_type in ("roberta", "xlm-roberta"):
            decoder.roberta.encoder.layer = decoder.roberta.encoder.layer[-num_decoder_layers:]
        else:
            raise ValueError(f"Unsupported model_type: {decoder_config.model_type}")

        decoder_config.num_hidden_layers = num_decoder_layers

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    config.tie_word_embeddings = False
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    processor = get_processor(encoder_name, decoder_name)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model, processor

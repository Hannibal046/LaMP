# import copy,math,random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from typing import Dict, List, Optional, Tuple, Union
# from transformers.activations import ACT2FN
# from transformers.modeling_utils import PreTrainedModel
# from transformers.modeling_outputs import (
#     BaseModelOutput,
#     BaseModelOutputWithPastAndCrossAttentions,
#     CausalLMOutputWithCrossAttentions,
#     Seq2SeqLMOutput,
#     Seq2SeqModelOutput,
# )
# from transformers.configuration_utils import PretrainedConfig
# from .module import *



# class DualEncoderTransformerConfig(PretrainedConfig):
#     model_type = "transformer"
#     keys_to_ignore_at_inference = ["past_key_values"]
#     attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

#     def __init__(
#         self,
#         vocab_size=6000,
#         max_position_embeddings=512,
#         encoder_layers=6,
#         encoder_ffn_dim=2048,
#         encoder_attention_heads=8,
#         decoder_layers=6,
#         decoder_ffn_dim=2048,
#         decoder_attention_heads=8,
#         encoder_layerdrop=0.0,
#         decoder_layerdrop=0.0,
#         use_cache=True,
#         is_encoder_decoder=True,
#         activation_function="relu",
#         d_model=512,
#         dropout=0.1,
#         attention_dropout=0.1,
#         activation_dropout=0.0,
#         init_std=0.02,
#         classifier_dropout=0.0,
#         scale_embedding=False,
#         decoder_start_token_id=0,
#         pad_token_id=4,
#         eos_token_id=1,
#         forced_eos_token_id=1,
#         bos_token_id=0,
#         positional_embedding_type='learned',
#         share_dual_encoder_parameters=True,
#         decoder_type='fid', #'dual_cross_attention',
#         **kwargs
#     ):
#         self.decoder_type = decoder_type
#         self.share_dual_encoder_parameters = share_dual_encoder_parameters
#         self.positional_embedding_type = positional_embedding_type
#         self.bos_token_id = bos_token_id
#         self.vocab_size = vocab_size
#         self.max_position_embeddings = max_position_embeddings
#         self.d_model = d_model
#         self.encoder_ffn_dim = encoder_ffn_dim
#         self.encoder_layers = encoder_layers
#         self.encoder_attention_heads = encoder_attention_heads
#         self.decoder_ffn_dim = decoder_ffn_dim
#         self.decoder_layers = decoder_layers
#         self.decoder_attention_heads = decoder_attention_heads
#         self.dropout = dropout
#         self.attention_dropout = attention_dropout
#         self.activation_dropout = activation_dropout
#         self.activation_function = activation_function
#         self.init_std = init_std
#         self.encoder_layerdrop = encoder_layerdrop
#         self.decoder_layerdrop = decoder_layerdrop
#         self.classifier_dropout = classifier_dropout
#         self.use_cache = use_cache
#         self.num_hidden_layers = encoder_layers
#         self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
#         super().__init__(
#             pad_token_id=pad_token_id,
#             eos_token_id=eos_token_id,
#             is_encoder_decoder=is_encoder_decoder,
#             decoder_start_token_id=decoder_start_token_id,
#             forced_eos_token_id=forced_eos_token_id,
#             **kwargs,
#         )

# class BaseTransformer(BasePreTrainedModel):
                  
#     def __init__(self, config):
#         super().__init__(config)

#         padding_idx, vocab_size = config.pad_token_id, config.vocab_size
#         self.decoder_type = config.decoder_type
#         # We always use self.shared for token embeddings to ensure compatibility with all marian models
#         self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
#         encoder_embed_tokens = decoder_embed_tokens = self.shared

#         self.encoder = DualEncoderTransformerEncoder(config, encoder_embed_tokens)
#         if config.decoder_type == 'fid':
#             self.decoder = TransformerDecoder(config, decoder_embed_tokens)
#         elif config.decoder_type == 'dual_cross_attention':
#             self.decoder = DualCrossAttnTransformerDecoder(config, decoder_embed_tokens)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         # This will return shared embeddings if they are shared else specific to encoder.
#         return self.get_encoder().get_input_embeddings()

#     def set_input_embeddings(self, value):
#         if self.config.share_encoder_decoder_embeddings:
#             self.shared = value
#             self.encoder.embed_tokens = self.shared
#             self.decoder.embed_tokens = self.shared
#         else:  # if not shared only set encoder embeedings
#             self.encoder.embed_tokens = value

#     def get_decoder_input_embeddings(self):
#         if self.config.share_encoder_decoder_embeddings:
#             raise ValueError(
#                 "`get_decoder_input_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
#                 "is `True`. Please use `get_input_embeddings` instead."
#             )
#         return self.get_decoder().get_input_embeddings()

#     def set_decoder_input_embeddings(self, value):
#         if self.config.share_encoder_decoder_embeddings:
#             raise ValueError(
#                 "`config.share_encoder_decoder_embeddings` is set to `True` meaning the decoder input embeddings "
#                 "are shared with the encoder. In order to set the decoder input embeddings, you should simply set "
#                 "the encoder input embeddings by calling `set_input_embeddings` with the appropriate embeddings."
#             )
#         self.decoder.embed_tokens = value

#     def get_encoder(self):
#         return self.encoder

#     def get_decoder(self):
#         return self.decoder

#     def resize_decoder_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         if self.config.share_encoder_decoder_embeddings:
#             raise ValueError(
#                 "`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
#                 "is `True`. Please use `resize_token_embeddings` instead."
#             )

#         old_embeddings = self.get_decoder_input_embeddings()
#         new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
#         self.set_decoder_input_embeddings(new_embeddings)

#         model_embeds = self.get_decoder_input_embeddings()

#         if new_num_tokens is None:
#             return model_embeds

#         # Update base model and current model config
#         self.config.decoder_vocab_size = new_num_tokens

#         # Tie weights again if needed
#         self.tie_weights()

#         return model_embeds

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         context_input_ids=None,
#         context_attention_mask=None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         decoder_head_mask: Optional[torch.Tensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Seq2SeqModelOutput:

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if encoder_outputs is None:
#             encoder_outputs = self.encoder(
#                 context_input_ids = context_input_ids,
#                 context_attention_mask = context_attention_mask,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#         if self.decoder_type == 'fid':
            
#             decoder_outputs = self.decoder(
#                 input_ids=decoder_input_ids,
#                 attention_mask=decoder_attention_mask,
#                 encoder_hidden_states=torch.cat((encoder_outputs.src_hidden_states,encoder_outputs.context_hidden_states),dim=1),
#                 encoder_attention_mask=torch.cat((attention_mask,context_attention_mask),dim=1),
#                 head_mask=decoder_head_mask,
#                 cross_attn_head_mask=cross_attn_head_mask,
#                 past_key_values=past_key_values,
#                 inputs_embeds=decoder_inputs_embeds,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#         elif self.decoder_type == 'dual_cross_attention':

#             decoder_outputs = self.decoder(
#                 input_ids=decoder_input_ids,
#                 attention_mask=decoder_attention_mask,
#                 encoder_hidden_states=encoder_outputs.src_hidden_states,
#                 encoder_attention_mask=attention_mask,
#                 context_hidden_states=encoder_outputs.context_hidden_states,
#                 context_attention_mask=context_attention_mask,
#                 head_mask=decoder_head_mask,
#                 cross_attn_head_mask=cross_attn_head_mask,
#                 past_key_values=past_key_values,
#                 inputs_embeds=decoder_inputs_embeds,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )

#         if not return_dict:
#             return decoder_outputs + encoder_outputs

#         return Seq2SeqModelOutput(
#             last_hidden_state=decoder_outputs.last_hidden_state,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )

# class DualEncoderTransformer(BasePreTrainedModel):
#     base_model_prefix = "model"
#     _keys_to_ignore_on_load_missing = [
#         r"final_logits_bias",
#         r"encoder.version",
#         r"decoder.version",
#         r"lm_head.weight",
#     ]

#     # _keys_to_ignore_on_save = ["model.encoder.embed_positions.weight", "model.decoder.embed_positions.weight"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = BaseTransformer(config)

#         target_vocab_size = config.vocab_size
#         self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
#         self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_encoder(self):
#         return self.model.get_encoder()

#     def get_decoder(self):
#         return self.model.get_decoder()

#     def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         new_embeddings = super().resize_token_embeddings(new_num_tokens)
#         if self.config.share_encoder_decoder_embeddings:
#             self._resize_final_logits_bias(new_num_tokens)
#         return new_embeddings

#     def _resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         old_embeddings = self.get_input_embeddings()
#         new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
#         self.set_input_embeddings(new_embeddings)

#         # update config.decoder_vocab_size if embeddings are tied
#         if self.config.share_encoder_decoder_embeddings:
#             self.config.decoder_vocab_size = new_num_tokens

#         # if word embeddings are not tied, make sure that lm head is resized as well
#         if (
#             self.config.share_encoder_decoder_embeddings
#             and self.get_output_embeddings() is not None
#             and not self.config.tie_word_embeddings
#         ):
#             old_lm_head = self.get_output_embeddings()
#             new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
#             self.set_output_embeddings(new_lm_head)

#         return self.get_input_embeddings()

#     def resize_decoder_token_embeddings(self, new_num_tokens):
#         if self.config.share_encoder_decoder_embeddings:
#             raise ValueError(
#                 "`resize_decoder_token_embeddings` should not be called if `config.share_encoder_decoder_embeddings` "
#                 "is `True`. Please use `resize_token_embeddings` instead."
#             )

#         old_embeddings = self.model.get_decoder_input_embeddings()
#         new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
#         self.model.set_decoder_input_embeddings(new_embeddings)

#         # if word embeddings are not tied, make sure that lm head is resized as well
#         if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
#             old_lm_head = self.get_output_embeddings()
#             new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
#             self.set_output_embeddings(new_lm_head)

#         model_embeds = self.model.get_decoder_input_embeddings()

#         if new_num_tokens is None:
#             return model_embeds

#         # Update base model and current model config
#         self.config.decoder_vocab_size = new_num_tokens

#         # Tie weights again if needed
#         self.tie_weights()

#         self._resize_final_logits_bias(new_num_tokens)

#         return model_embeds

#     def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
#         old_num_tokens = self.final_logits_bias.shape[-1]
#         if new_num_tokens <= old_num_tokens:
#             new_bias = self.final_logits_bias[:, :new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
#             new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
#         self.register_buffer("final_logits_bias", new_bias)

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings: nn.Embedding):
#         self.lm_head = new_embeddings

#     def tie_weights(self):
#         """
#         Tie the weights between the input embeddings and the output embeddings.

#         If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
#         weights instead.
#         """
#         output_embeddings = self.get_output_embeddings()
#         if output_embeddings is not None and getattr(self.config, "tie_word_embeddings", True):
#             # if embeddings are shared this will return shared embeddings otherwise decoder embed_tokens
#             word_embeddings = self.get_decoder().get_input_embeddings()
#             self._tie_or_clone_weights(output_embeddings, word_embeddings)

#         if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
#             if hasattr(self, self.base_model_prefix):
#                 self = getattr(self, self.base_model_prefix)
#             self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

#         for module in self.modules():
#             if hasattr(module, "_tie_weights"):
#                 module._tie_weights()

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         context_input_ids=None,
#         context_attention_mask=None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         decoder_head_mask: Optional[torch.Tensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Seq2SeqLMOutput:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Returns:

#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if labels is not None:
#             if use_cache:
#                 logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
#             use_cache = False
#             if decoder_input_ids is None and decoder_inputs_embeds is None:
#                 decoder_input_ids = shift_tokens_right(
#                     labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                 )

#         outputs = self.model(
#             input_ids,
#             context_input_ids=context_input_ids,
#             context_attention_mask=context_attention_mask,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=encoder_outputs,
#             decoder_attention_mask=decoder_attention_mask,
#             head_mask=head_mask,
#             decoder_head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             decoder_inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.decoder_vocab_size), labels.view(-1))

#         if not return_dict:
#             output = (lm_logits,) + outputs[1:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

#         return Seq2SeqLMOutput(
#             loss=masked_lm_loss,
#             logits=lm_logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )

#     def prepare_inputs_for_generation(
#         self,
#         decoder_input_ids: torch.LongTensor,
#         past: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         decoder_head_mask: Optional[torch.Tensor] = None,
#         context_attention_mask=None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         use_cache: Optional[bool] = None,
#         encoder_outputs: Optional[Union[Tuple[torch.Tensor], BaseModelOutput]] = None,
#         **kwargs,
#     ) -> Dict:
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             decoder_input_ids = decoder_input_ids[:, -1:]

#         return {
#             "input_ids": None,  # encoder_outputs is defined. input_ids not needed
#             "encoder_outputs": encoder_outputs,
#             "past_key_values": past,
#             "decoder_input_ids": decoder_input_ids,
#             "attention_mask": attention_mask,
#             "head_mask": head_mask,
#             "context_attention_mask":context_attention_mask,
#             "decoder_head_mask": decoder_head_mask,
#             "cross_attn_head_mask": cross_attn_head_mask,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#         }

#     def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
#         return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

#     def adjust_logits_during_generation(self, logits, cur_len):
#         logits[:, self.config.pad_token_id] = float("-inf")  # never predict pad token.
#         return logits

#     @staticmethod
#     def _reorder_cache(past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             # cached cross_attention states don't have to be reordered -> they are always the same
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
#             )
#         return reordered_past

#     @staticmethod
#     def _expand_inputs_for_generation(
#         input_ids: torch.LongTensor,
#         expand_size: int = 1,
#         is_encoder_decoder: bool = False,
#         attention_mask: Optional[torch.LongTensor] = None,
#         context_attention_mask=None,
#         encoder_outputs  = None,
#         **model_kwargs,
#     ):
#         expanded_return_idx = (
#             torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
#         )
#         input_ids = input_ids.index_select(0, expanded_return_idx)

#         if "token_type_ids" in model_kwargs:
#             token_type_ids = model_kwargs["token_type_ids"]
#             model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

#         if attention_mask is not None:
#             # print(attention_mask)
#             model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
        
#         if context_attention_mask is not None:
#             # print(attention_mask)
#             model_kwargs["context_attention_mask"] = context_attention_mask.index_select(0, expanded_return_idx)

#         if is_encoder_decoder:
#             if encoder_outputs is None:
#                 raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
#             # encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
#             #     0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
#             # )
#             encoder_outputs["src_hidden_states"] = encoder_outputs.src_hidden_states.index_select(
#                 0, expanded_return_idx.to(encoder_outputs.src_hidden_states.device)
#             )
#             encoder_outputs["context_hidden_states"] = encoder_outputs.context_hidden_states.index_select(
#                 0, expanded_return_idx.to(encoder_outputs.context_hidden_states.device)
#             )
#             model_kwargs["encoder_outputs"] = encoder_outputs
#         return input_ids, model_kwargs

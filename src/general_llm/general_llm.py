from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM, GPT2LMHeadModel, LlamaForCausalLM, AutoTokenizer

from src.dataset import build_retrieval_input


class GeneralLLMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def loss_caption(self, logits, captioning_ids: List[int], targets_ids: torch.LongTensor) -> torch.FloatTensor:
        return torch.nn.functional.cross_entropy(
            logits[captioning_ids, :-1].reshape(-1, logits.size(-1)), 
            targets_ids[captioning_ids].flatten(), 
            ignore_index=0
        )

    def loss_embedding(
        self, 
        logits, 
        pred_clip_embeds: torch.FloatTensor, 
        clip_embeds: torch.FloatTensor, 
        target_clip_ids: torch.LongTensor, 
        embedding_ids: List[int]
    ) -> torch.FloatTensor:
        ce_loss = self.ce(
            logits[embedding_ids, :-1].view(-1, logits.size(-1)), 
            target_clip_ids[embedding_ids, 1:].view(-1)
        )

        mse_loss = self.mse(
            pred_clip_embeds, 
            clip_embeds[embedding_ids]
        )

        return mse_loss + ce_loss

    def forward(
        self,
        logits,
        clip_embeds: torch.FloatTensor,
        target_ids: torch.LongTensor,
        pred_clip_embeds: torch.FloatTensor,
        target_clip_ids: torch.LongTensor,
        embedding_ids: List[int],
        captioning_ids: List[int]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        loss_embedding = 0
        loss_caption = 0

        if len(embedding_ids) > 0:
            loss_embedding += self.loss_embedding(logits, pred_clip_embeds, clip_embeds, target_clip_ids, embedding_ids)
        
        if len(captioning_ids) > 0:
            loss_caption += self.loss_caption(logits, captioning_ids, target_ids)

        loss = loss_embedding + loss_caption

        return loss, loss_embedding, loss_caption

class GeneralLLMPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        pass

class GeneralLLM(GeneralLLMPreTrainedModel):
    def __init__(self, llm: AutoModelForCausalLM, clip_dim: int = 768) -> None:
        super().__init__(llm.config)
        self.llm = llm
        self.clip_dim = clip_dim

        if self.llm.config.hidden_size != self.clip_dim:
            self.clip_projection = nn.Linear(self.clip_dim, self.llm.config.hidden_size)
        else: 
            self.clip_projection = nn.Identity()

    def token_embeder(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if isinstance(self.llm, GPT2LMHeadModel):
            token_embeder = self.llm.transformer.wte
        elif isinstance(self.llm, LlamaForCausalLM):
            token_embeder = self.llm.model.embed_tokens
        else:
            raise ValueError(f"Expected model to be of type GPT2LMHeadModel or LlamaForCausalLM but got {type(self.llm)}")
        
        return token_embeder(input_ids)

    def forward(self, **kwargs):
        return self.llm(**kwargs)

    # Adapted from https://github.com/rmokady/CLIP_prefix_caption/blob/main/predict.py
    def generate_caption(
        self,
        embed: torch.FloatTensor,
        stop_token_index: int,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.0,
    ) -> List[int]:
        filter_value = -float("Inf")
        tokens = None

        with torch.no_grad():
            for entry_idx in range(entry_count):
                generated = embed

                for i in range(entry_length):
                    outputs = self(inputs_embeds=generated)

                    logits = outputs.logits
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value

                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                    next_token_embed = self.token_embeder(next_token)

                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)

                    generated = torch.cat((generated, next_token_embed), dim=1)

                    if stop_token_index == next_token.item():
                        break

                output_list = list(tokens.squeeze().cpu().numpy())

        return output_list

    def generate_clip_embedding(
            self, 
            input_ids: torch.LongTensor, 
            attention_mask: torch.LongTensor
        ) -> torch.FloatTensor:
        caption = build_retrieval_input(caption, False)
        outputs = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True, 
            output_hidden_states=True, 
            max_new_tokens=50
        )

        return outputs['hidden_states'][-3][-1][:,-1,:]
        
def calculate_token_importance_perturbation(
    token_ids_reasoning,
    token_ids_before_reasoning,
    token_ids_conclude,
    target_pfinal,
    model,
    pad_token_id=PAD_TOKEN_ID,
    pfinal_mode = 'last',
    answer_token_ids = None
):
    R = len(token_ids_reasoning)
    batch_inputs = []
    for mask_pos in range(R):
        masked_reasoning = token_ids_reasoning.copy()
        masked_reasoning[mask_pos] = pad_token_id

        seq = token_ids_before_reasoning + masked_reasoning + token_ids_conclude
        batch_inputs.append(seq)

    max_len = max(len(x) for x in batch_inputs)
    padded = []
    for x in batch_inputs:
        padded.append(x + [pad_token_id] * (max_len - len(x)))

    input_tensor = torch.tensor(padded).to(model.device)
    with torch.no_grad():
        out = model(input_tensor)
        logits_batch = out.logits.float()  # [B, L, V]

    kl_or_delta_list = []

    if pfinal_mode in ["last", "answer_mean"]:
        target_logits = target_pfinal  # [V]
        t_log_softmax = F.log_softmax(target_logits, dim=-1)
        t_softmax = F.softmax(target_logits, dim=-1)

        for i in range(R):
            if pfinal_mode == "last":
                masked_logits = logits_batch[i, -1, :]  # [V]
            elif pfinal_mode == "answer_mean":
                masked_logits = mean_answer_logits_from_full_logits(
                    logits_batch[i], answer_token_ids
                )  # [V]

            logQ = F.log_softmax(masked_logits, dim=-1)
            KL = torch.sum(t_softmax * (t_log_softmax - logQ))
            kl_or_delta_list.append(KL.item())

    elif pfinal_mode == "answer_joint":
        baseline_score = target_pfinal  # scalar tensor

        for i in range(R):
            masked_score = joint_logp_from_full_logits(
                logits_batch[i],  # [L,V]
                full_input_ids=padded[i],
                answer_token_ids=answer_token_ids
            )  # scalar
            importance = (baseline_score - masked_score).item()
            kl_or_delta_list.append(importance)

    return kl_or_delta_list

def calculate_token_importance_gradient(
    model,
    input_segments,
    device,
    pfinal_mode="answer_mean",
    noise_std=1e-3,
    eps=1e-12,
):
    import torch
    import torch.nn.functional as F

    ids_before   = input_segments["token_ids_before_reasoning"]
    ids_reason   = input_segments["token_ids_reasoning"]
    ids_conclude = input_segments.get("token_ids_conclude", [])
    ids_answer   = input_segments.get("token_ids_answer", [])

    full_ids = ids_before + ids_reason + ids_conclude + ids_answer
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)

    start_r = len(ids_before)
    end_r = start_r + len(ids_reason)

    def pfinal_logits_from_full_logits(full_logits_2d):
        L, V = full_logits_2d.shape
        if pfinal_mode == "last":
            return full_logits_2d[-1, :]
        elif pfinal_mode == "answer_mean":
            A = len(ids_answer)
            start = L - A
            end = L
            pred_positions = [p for p in range(start-1, end-1) if 0 <= p < L]
            return full_logits_2d[pred_positions, :].mean(dim=0)
        else:
            raise ValueError("For this noisy-KL version, use pfinal_mode in {'last','answer_mean'}")

    def answer_joint_phi_from_full_logits(full_logits_2d):
        L, V = full_logits_2d.shape
        A = len(ids_answer)
        if A == 0:
            return full_logits_2d.new_tensor(0.0)

        ans_token_pos = list(range(L - A, L))         
        pred_pos      = [p - 1 for p in ans_token_pos]  
        pred_pos      = [p for p in pred_pos if 0 <= p < L]

        offset = (len(ans_token_pos) - len(pred_pos))
        ans_token_ids = ids_answer[offset:]  

        logits = full_logits_2d[pred_pos, :]                 # [K,V]
        logprobs = F.log_softmax(logits, dim=-1)             # [K,V]
        target = torch.tensor(ans_token_ids, device=logits.device, dtype=torch.long)  # [K]
        token_ll = logprobs.gather(dim=-1, index=target[:, None]).squeeze(-1)         # [K]

        phi = token_ll.mean()  # scalar
        return phi

    with torch.no_grad():
        out0 = model(input_ids=input_ids)
        full_logits0 = out0.logits[0].float()  # [L,V]

        if pfinal_mode in ("last", "answer_mean"):
            p0_logits = pfinal_logits_from_full_logits(full_logits0)
            p0 = F.softmax(p0_logits, dim=-1).detach()  # distribution teacher
        elif pfinal_mode == "answer_joint":
            phi0 = answer_joint_phi_from_full_logits(full_logits0).detach()  # scalar teacher
        else:
            raise ValueError("pfinal_mode must be {'last','answer_mean','answer_joint'}")


    emb_layer = model.get_input_embeddings()
    inputs_embeds = emb_layer(input_ids).detach()
    inputs_embeds.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    noise = torch.zeros_like(inputs_embeds)
    noise[:, start_r:end_r, :].normal_(mean=0.0, std=noise_std)
    noisy_embeds = inputs_embeds + noise

    out = model(inputs_embeds=noisy_embeds)
    full_logits = out.logits[0].float()  # [L,V]

    if pfinal_mode in ("last", "answer_mean"):
        p_logits = pfinal_logits_from_full_logits(full_logits)     # [V]
        logp = F.log_softmax(p_logits, dim=-1)                     # log p_noisy(v)
        J = torch.sum(p0 * (torch.log(p0 + eps) - logp))
    elif pfinal_mode == "answer_joint":
        phi_noisy = answer_joint_phi_from_full_logits(full_logits)
        J = (phi0 - phi_noisy)
    else:
        raise ValueError

    J.backward()

    grads = inputs_embeds.grad[0]  # [L,H]
    reasoning_grads = grads[start_r:end_r]
    reasoning_embeds = inputs_embeds[0, start_r:end_r].detach()

    saliency = (reasoning_grads * reasoning_embeds).sum(dim=-1).abs()
    return saliency.detach().tolist()

def calculate_oar_hybrid_weights(self, inputs, model):
    threshold = self.args.threshold
    gating_beta = self.args.beta
    
    weight_tensor_list = []
    
    for i, reasoning in enumerate(inputs['reasoning']):
        input_case = create_test_case(
            inputs['intro'], 
            inputs['question'], 
            inputs['bridge'], 
            reasoning, 
            inputs['conclude'],
            inputs['answer']
        )
        input_segments = prepare_input_segments(input_case, self.tokenizer)
        
        if self.args.method == 'gradient':
            raw_scores = calculate_token_importance_gradient(
                model, 
                self.tokenizer, 
                input_segments, 
                self.args.device
            )
        elif self.args.method == 'perturbation':
            pfinal_mode = self.args.pfinal_mode 
            target_pfinal = get_pfinal_representation(
                model,
                input_segments['full_input_ids'],
                ids_answer=input_segments['token_ids_answer'],
                pfinal_mode=pfinal_mode
            )

            raw_scores = calculate_token_importance_perturbation(
                input_segments['token_ids_reasoning'],
                input_segments['token_ids_before_reasoning'],
                input_segments['token_ids_conclude'],
                target_pfinal,
                model,
                pad_token_id=PAD_TOKEN_ID,
                pfinal_mode=pfinal_mode,
                answer_token_ids=input_segments['token_ids_answer']
            )

        score_arr = np.array(raw_scores)
        log_score = np.log1p(score_arr)

        min_val = log_score.min()
        max_val = log_score.max()
        range_val = max_val - min_val if max_val > min_val else 1.0

        norm_importance = (log_score - min_val) / range_val

        threshold = np.quantile(norm_importance, threshold)
        weights = np.ones_like(norm_importance)

        # Noise suppression
        mask_low = norm_importance < threshold
        weights[mask_low] = norm_importance[mask_low] / (threshold + 1e-8)
        weights[mask_low] = np.maximum(weights[mask_low], 0.1)

        # Signal boosting
        mask_high = ~mask_low
        if mask_high.any():
            relative_pos = (norm_importance[mask_high] - threshold) / (1.0 - threshold + 1e-8)
            weights[mask_high] = 1.0 + gating_beta * relative_pos
        
        if self.args.sum_to_L:
            L = weights.shape[0]
            denom = weights.sum()
            if denom <= 0:
                weights = np.ones_like(weights)
            else:
                weights = weights * (L / (denom + 1e-8))
        
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.args.device)
        weight_tensor_list.append(weight_tensor)
        
    return weight_tensor_list

def compute_loss(self, model, inputs):
    
    prompt_response_ids = inputs['prompt_response_ids']
    attention_mask = inputs['attention_mask']
    action_mask = inputs['action_mask']
    num_actions = action_mask.size(1)
    action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
    
    advantages = inputs['advantages'].unsqueeze(1)

    if self.enable_credit_assignment:
        weights_list = self.calculate_oar_hybrid_weights(inputs, model)
        padded_weights_list = []
        
        for i, w in enumerate(weights_list):
            
            full_seq_weight = torch.ones(num_actions, device=self.args.device)
            
            res_len = w.size(0)
            
            if res_len <= num_actions:
                full_seq_weight[:res_len] = w
            else:
                full_seq_weight = w[:num_actions]
            
            padded_weights_list.append(full_seq_weight)
        
        weight_matrix = torch.stack(padded_weights_list, dim=0)
        
        advantages = advantages * weight_matrix

    old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
    coef_1 = torch.exp(action_log_probs - old_action_log_probs) 
    coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
    per_token_loss1 = coef_1 * advantages 
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    per_token_loss = per_token_loss * action_mask

    loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1) # [batch_size * num_generations]
    loss = loss.mean()

    return loss

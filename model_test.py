def ProcrustEs(self, head, relation, tail, mode):
    def get_score(head_batch, rel_batch, tail_batch, eps=1e-7):
        score_batch = None
        for i in range(0, head_batch.size(0)): # iter over mini-batch
            head_emb = torch.stack(head_batch[i].chunk(self.td // self.sd, dim=1))
            # each mini-batch is for a sub_emb 
            rot_emb = rel_batch[i].view(self.td // self.sd, self.sd, self.sd)
            out_emb = head_emb.bmm(rot_emb)
            out_emb = torch.cat(out_emb.split(1), 2).view(head.size(1), self.td)
            tail_emb = tail_batch[i]
            score = - (out_emb - tail_emb).norm(dim=1).view(1, head.size(1))
            if i == 0:
                score_batch = torch.cat((score,), 0)
            else:               
                score_batch = torch.cat((score_batch, score), 0)
        return score_batch

    if mode == 'head-batch': # select best head -> find the head whose map is closest to tail
        tail = tail.repeat(1, head.size(1), 1)
    else: # select best tail
        head = head.repeat(1, tail.size(1), 1)
    return get_score(head, relation, tail)
    
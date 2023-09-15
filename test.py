### Exac
for epoch in epochs:
    Generator.train()
    Discriminator.eval()
    for epoch in epoch_G_preset:
        data = get_batch()
        optimizer.zero_grad()
        result = Generator(data)
        
        loss = Generator_loss(result, ground_truth_G)
        loss.backward()
        optimizer.step()

    Discriminator.train()
    Generator.eval()
    for epoch in epoch_D:
        classification = Discriminator(result)
        D_margin_fake = D_fake_scores - D_neg_scores
        D_fake_loss = F.relu(margin - D_margin_fake).mean()
        loss = Discriminator_loss(classification, ground_truth_D)
        loss.backward()
        optimizer.step()
        
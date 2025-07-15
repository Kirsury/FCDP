from Model import *

model = DefectPredictionModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for batch in dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    logits = model(input_ids, attention_mask)
    loss = compute_loss(logits, labels)
    metrics = compute_metrics(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

from torch import optim

# Train the BNN model
def train_model(model, train_dataloader, epochs=1000, lr=0.001):
    """
    Train the BNN model using minibatching via DataLoader.
    
    Args:
        model: The BNN model to train
        train_dataloader: DataLoader containing (x, y) pairs
        epochs (int): Number of training epochs
        lr (float): Learning rate
    
    Returns:
        tuple: (trained_model, losses)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Iterate through minibatches
        for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
            # Forward pass - compute negative ELBO for this batch
            batch_loss = -model.elbo(x_batch, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # Accumulate loss for this epoch
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        # Average loss across all batches for this epoch
        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)
        
        # Print progress
        if (epoch + 1) % 200 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}')
    
    return model, losses
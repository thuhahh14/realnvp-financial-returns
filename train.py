import torch


def train_realnvp(model, X_train_tensor, X_test_tensor, epochs=300, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    model.to(device)
    X_train_tensor = X_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)

    for epoch in range(epochs):
        model.train()
        loss = -model.log_prob(X_train_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = -model.log_prob(X_test_tensor).mean()

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train NLL: {loss.item():.6f} | "
                f"Test NLL: {test_loss.item():.6f}"
            )

    return train_losses, test_losses

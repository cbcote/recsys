import numpy as np


class SDGMF:
    def __init__(self, k=5, learning_rate=0.01, regularization=0.01, epochs=100) -> None:
        self.k = k
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
    
    def fit(self, X, X_val=None):
        num_users, num_items = X.shape
        self.users = np.random.normal(scale=1./self.k, size=(num_users, self.k))
        self.items = np.random.normal(scale=1./self.k, size=(num_items, self.k))
        
        # Early stopping initialization if validation data is provided
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(self.epochs):
            error_sum = 0
            for u in range(num_users):
                for i in range(num_items):
                    if X[u, i] > 0:
                        error = X[u, i] - self.predict(u, i)
                        self.users[u] += self.learning_rate * (error * self.items[i] - self.regularization * self.users[u])
                        self.items[i] += self.learning_rate * (error * self.users[u] - self.regularization * self.items[i])
                        # Sum the errors to compute the loss for each epoch and monitor the training process
                        error_sum += error**2
            
            self.losses.append(error_sum)

           # Validation and early stopping check
            if X_val is not None:
                val_loss = ((X_val - self.predict_all())**2).sum()
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > 5:  # For example, stop if no improvement after 5 epochs
                        print(f"Early stopping at epoch {epoch}")
                        return
            
            print(f"Epoch {epoch}: Training Loss = {error_sum}")  # For monitoring
        
    def predict(self, u, i):
        return self.users[u].dot(self.items[i].T)
    
    def predict_all(self):
        return self.users.dot(self.items.T)
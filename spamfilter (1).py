import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Lets Load the data

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
data = pd.read_csv(data_url, header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# We need to preprocess the data to standardize the features
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y , test_size=0.25, random_state=42)

# Lets define the datasets and dataloader

class EmailDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float), torch.tensor(self.Y[index], dtype=torch.float)

train_dataset = EmailDataset(X_train, y_train)
test_dataset = EmailDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Time to create a Neural Network Model

class SpamClassifier(torch.nn.Module):
  def __init__(self, input_dim):
      super(SpamClassifier, self).__init__()
      self.layer = torch.nn.Sequential(
          torch.nn.Linear(input_dim, 64),
          torch.nn.ReLU(),
          torch.nn.Linear(64, 1),
          torch.nn.Sigmoid()
      )

  def forward(self, x):
      return self.layer(x).squeeze()

# Lets train the model:

model = SpamClassifier(X_train.shape[1])
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 46

for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    # Evaluation on Test set
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            outputs = model(data)
            predicted = (outputs > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"Epoch {epoch+1}/{epochs}, Accuracy: {correct/total:.2f}")

# lets test our model on our own data
def mock_transform(emails, feature_length):
    return scaler.transform(np.random.rand(len(emails), feature_length))


test_emails = [
    "You have won a lottery!",
    "Your facebook password is changed",
    "Dear User, Please click on the link below",
    "Regarding you application in the college"
]

transformed_emails = mock_transform(test_emails, X_train.shape[1])

def predict_spam(emails_transformed, trained_model):
    email_tensor = torch.tensor(emails_transformed, dtype=torch.float)
    outputs = trained_model(email_tensor).squeeze()
    predictions = (outputs > 0.5).float().numpy()
    return predictions

predictions = predict_spam(transformed_emails, model)

for email, pred in zip(test_emails, predictions):
  result = "SPAM" if pred == 1 else "NOT SPAM"
  print(f"Email: {email[:50]}... - Prediction: {result}")

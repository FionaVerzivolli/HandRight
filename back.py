import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, load
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

training_data = datasets.EMNIST(
    root="data",
    split='byclass',
    train=True,
    download=True,
    transform=ToTensor(),
)

#create dataloader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 62)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

p = transforms.Compose([transforms.Resize((28, 28))])


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def response(canvas_path, expected_char):
    """
    Predicts the character from the canvas image and compares it to the expected character,
    returning a confidence score for the prediction.
    """

    with open('image_recognition.pth', 'rb') as f:
        model.load_state_dict(load(f, map_location=device))


    canvas_img = Image.open(canvas_path).convert('L')
    canvas_img = p(canvas_img)  # Apply the transformations
    
    #tensor conversion
    canvas_tensor = ToTensor()(canvas_img).unsqueeze(0).to(device)

    #get prediction
    model.eval()
    with torch.no_grad():
        canvas_logits = model(canvas_tensor)
        canvas_probs = F.softmax(canvas_logits, dim=1)
        canvas_pred_index = torch.argmax(canvas_probs, dim=1).item()
        confidence_score = torch.max(canvas_probs).item()

    
    if canvas_pred_index < 10:
        
        canvas_pred_char = str(canvas_pred_index)
    elif 10 <= canvas_pred_index < 36:
        
        canvas_pred_char = chr(canvas_pred_index + 55)
    else:
       
        canvas_pred_char = chr(canvas_pred_index + 61)

   
    print(f"Canvas predicted character: {canvas_pred_char}")
    print(f"Expected character: {expected_char}")
    print(f"Confidence score of the prediction: {confidence_score:.2f}")

   
    score = confidence_score if canvas_pred_char == expected_char else 0

    return [canvas_pred_char, score]


if __name__ == "__main__":
    pass
    # epochs = 10
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    # print("Done!")
    # torch.save(model.state_dict(), 'image_recognition.pth')

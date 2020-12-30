import os

model_type = 'image_cnn'

if model_type == 'rnn' or model_type == '3dcnn':
    import torch
    import torch.nn as nn
    from torchvision import models
    import torchvision.transforms as transforms

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    class Resnt18Rnn(nn.Module):
        def __init__(self, params_model):
            super(Resnt18Rnn, self).__init__()
            num_classes = params_model["num_classes"]
            dropout = params_model["dropout"]
            pretrained = params_model["pretrained"]
            rnn_hidden_size = params_model["rnn_hidden_size"]
            rnn_num_layers = params_model["rnn_num_layers"]

            baseModel = models.resnet18(pretrained=pretrained)
            num_features = baseModel.fc.in_features
            baseModel.fc = Identity()
            self.baseModel = baseModel
            self.dropout = nn.Dropout(dropout)
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
            self.fc1 = nn.Linear(rnn_hidden_size, num_classes)

        def forward(self, x):
            b_z, ts, c, h, w = x.shape
            ii = 0
            y = self.baseModel((x[:, ii]))
            output, (hn, cn) = self.rnn(y.unsqueeze(1))
            for ii in range(1, ts):
                y = self.baseModel((x[:, ii]))
                out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
            out = self.dropout(out[:, -1])
            out = self.fc1(out)
            return out

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    model_path = os.path.join(
        'models', 'convolution-rnn-18-2-512-fully-trained.pt' if model_type == 'rnn' else '3dcnn-18-equalised.pt')

    timesteps = 8
    if model_type == "rnn":
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        h, w = 224, 224
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

    transformer = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if model_type == "rnn":
        params_model = {
            "num_classes": 4,
            "dropout": 0.3,
            "pretrained": False,
            "rnn_num_layers": 2,
            "rnn_hidden_size": 512, }
        model = Resnt18Rnn(params_model)
    else:
        model = models.video.r3d_18(pretrained=True, progress=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 4)

    model.load_state_dict(torch.load(model_path, map_location=device))

    def get_engagement_level(video_frames):
        transformed_images = list(map(transformer, video_frames))
        if model_type == 'rnn':
            transformed_images = torch.stack(transformed_images).unsqueeze(0)
        else:
            transformed_images = torch.stack(transformed_images)
        output = model(transformed_images)
        return torch.argmax(output).data/4
else:
    from fastai.vision import load_learner, open_image
    model = load_learner('models', 'resnet18-stage-1-equalised.pkl')

    def get_engagement_level(image_path):
        _, pred_class, _ = model.predict(open_image(image_path))
        return pred_class/4

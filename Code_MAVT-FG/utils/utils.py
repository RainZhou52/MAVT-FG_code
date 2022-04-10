import sys
import torch
import torchmetrics
from tqdm import tqdm
import torch.nn.functional as F

def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_weight=1.):
    recall = torchmetrics.Recall(average='none', num_classes=10)
    precision = torchmetrics.Precision(average='none', num_classes=10)
    f1 = torchmetrics.F1Score(average='none', num_classes=10)

    recall.to(device)
    precision.to(device)
    f1.to(device)

    model.train()
    loss_function1 = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, images_data in enumerate(data_loader):
        images, images_labels, audio, audio_labels = images_data
        sample_num += images.shape[0]

        pred, cls_av, cls_v, cls_a = model(images.to(device), audio.to(device))

        print(cls_av.shape)

        recall(pred.argmax(1), images_labels.to(device))
        precision(pred.argmax(1), images_labels.to(device))
        f1(pred.argmax(1), images_labels.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, images_labels.to(device)).sum()

        loss = loss_function1(pred, images_labels.to(device)) + con_loss(cls_av, images_labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    total_recall = recall.compute()
    total_precision = precision.compute()
    total_f1 = f1.compute()

    print("recall of every train dataset class: ", total_recall.mean())
    print("precision of every train dataset class: ", total_precision.mean())
    print("F1score of every train dataset class: ", total_f1.mean())

    recall.reset()
    precision.reset()
    f1.reset()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, loss_weight=1.):
    recall = torchmetrics.Recall(average='none', num_classes=10)
    precision = torchmetrics.Precision(average='none', num_classes=10)
    f1 = torchmetrics.F1Score(average='none', num_classes=10)

    recall.to(device)
    precision.to(device)
    f1.to(device)

    loss_function1 = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, images_data in enumerate(data_loader):
        images, images_labels, audio, audio_labels = images_data
        sample_num += images.shape[0]

        pred, cls_av, cls_v, cls_a = model(images.to(device), audio.to(device))

        recall(pred.argmax(1), images_labels.to(device))
        precision(pred.argmax(1), images_labels.to(device))
        f1(pred.argmax(1), images_labels.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, images_labels.to(device)).sum()

        loss = loss_function1(pred, images_labels.to(device)) + con_loss(cls_av, images_labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    # Recall  Precision   F1-Score
    total_recall = recall.compute()
    total_precision = precision.compute()
    total_f1 = f1.compute()

    print("recall of every test dataset class: ", total_recall.mean())
    print("precision of every test dataset class: ", total_precision.mean())
    print("F1score of every test dataset class: ", total_f1.mean())

    recall.reset()
    precision.reset()
    f1.reset()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

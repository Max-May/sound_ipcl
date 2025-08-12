import torch
import torch.nn as nn

from dataloader.dataset_functions import Transform


def knn_monitor(model, trainFeatures, trainLabels, val_loader,
                K=200, sigma=1., n_samples=5, hrtf=None, num_chunks=200, device='cpu'):
    trainFeatures = trainFeatures.to(device)
    trainLabels = trainLabels.to(device)

    testFeatures, testLabels = get_features(model, val_loader, n_samples=n_samples, hrtf=hrtf, device=device)

    testFeatures = testFeatures.to(device)
    testLabels = testLabels.to(device)

    # make sure all the features are l2-normalized
    trainFeatures = nn.functional.normalize(trainFeatures, p=2, dim=1)
    testFeatures = nn.functional.normalize(testFeatures, p=2, dim=1)

    # split test features into chunks to avoid out-of-memory error:
    chunkFeatures = torch.chunk(testFeatures, num_chunks, dim=0)
    chunkLabels = torch.chunk(testLabels, num_chunks, dim=0)

    C = trainLabels.max() + 1
    top1, top5, total = 0., 0., 0.
    # for features, labels in progress_bar(zip(chunkFeatures, chunkLabels), total=num_chunks):
    for features, labels in zip(chunkFeatures, chunkLabels):
        top1_, top5_, total_ = do_kNN(trainFeatures, trainLabels, features, labels, C, K, sigma, device=device)
        top1 += top1_ / 100 * total_
        top5 += top5_ / 100 * total_
        total += total_
    top1 = top1 / total * 100
    top5 = top5 / total * 100
    
    print(f"run_kNN accuracy: top1={top1}, top5={top5}")
    return top1, top5


@torch.no_grad()
def get_features(model, loader, n_samples, hrtf, device='cpu', dtype=torch.float):
    features, labels = [], []
    transform = Transform(n_samples=n_samples, hrtf=hrtf, target_samplerate=48000)

    model.eval()
    for idx, (data, _) in enumerate(loader):
        # print(f'[{idx}/5]')
        data, targets = transform(data)
        data = data.to(device, dtype=dtype)
        targets = targets.to(device)
        out = model(data)

        features.append(out.view(out.shape[0],-1).to(device, non_blocking=True))
        labels.append(targets)

        # if idx == 5:
        #     break

    features = torch.cat(features)
    labels = torch.cat(labels)

    return features, labels


@torch.no_grad()
def gen_features(model, dataloader, n_samples, hrtf, device='cpu', dtype=torch.float):    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = Transform(n_samples=n_samples, hrtf=hrtf, target_samplerate=48000)

    model.eval()
    for idx, (data,_) in enumerate(dataloader):
        data, targets = transform(data)
        data = data.to(device, dtype=dtype)
        targets = targets.to(device)
        out = model(data)

        yield out.to(device), targets.to(device)
    


def do_kNN(trainFeatures, trainLabels, testFeatures, testLabels, C, K, sigma, device='cpu'):
    '''
        trainFeatures: [nTrainSamples, nFeatures]
        trainLabels: [nTrainSamples]
        
        testFeatures: [nTestSamples, nFeatures]
        testLabels: [nTestSamples]
    '''      
    batchSize = len(testLabels)

    dist = torch.mm(testFeatures, trainFeatures.T).to(device, non_blocking=True)
    
    yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
    
    candidates = trainLabels.view(1,-1).expand(batchSize, -1).to(device, non_blocking=True)
    
    retrieval = torch.gather(candidates, 1, yi)
    retrieval_one_hot = torch.zeros(K, C).to('cpu')
    retrieval_one_hot.resize_(batchSize * K, C).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1).cpu(), 1)
    yd_transform = yd.clone().div_(sigma).exp_()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1).cpu()), 1)
    _, predictions = probs.sort(1, True)
    
    # Find which predictions match the target
    correct = predictions.eq(testLabels.view(-1,1).cpu())
    # correct.shape
    
    total = correct.size(0)
    top1 = correct.narrow(1,0,1).sum().item() / total * 100
    top5 = correct.narrow(1,0,5).sum().item() / total * 100
    
    return top1, top5, total


def run_kNN_chunky(model, train_loader, test_loader, n_samples, hrtf, epoch_size, K=200, sigma=.07, num_chunks=10, device=None):
    '''this version scales better to larger feature spaces
    
        we compute the full testFeatures, testLabels,
        
        then we iterate over the training set in batches, accumulating `num_chunks` (should
        be `num_batches`, but keeping the naming the same as run_kNN for api consistency).
        
    '''
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    print("extracting test features...")
    testFeatures, testLabels = get_features(model, test_loader, n_samples=n_samples, hrtf=hrtf, device=device)
    C = testLabels.max() + 1
    
    print("extracting/comparing to train features...")
    dist = None
    trainLabels = None
    trainIndexes = None
    
    #for batch_num, (trainFeatures, labels, indexes) in enumerate(gen_features(model, train_loader, layer_name)):
    trainFeatures, labels, indexes = None, None, None
    for batch_num, (feat, lab) in enumerate(gen_features(model, train_loader, n_samples=n_samples, hrtf=hrtf, device=device)): 
        trainFeatures = feat if trainFeatures is None else torch.cat([trainFeatures, feat])
        labels = lab if labels is None else torch.cat([labels, lab])
        # indexes = ind if indexes is None else torch.cat([indexes, ind])
        
        # accumulate a few batches before proceeding
        if (batch_num+1) % num_chunks != 0 and batch_num != (epoch_size-1):
            continue
        
        # compute distances, concat with retained
        d = torch.mm(testFeatures, trainFeatures.T).to(device)        
        dist = d if dist is None else torch.cat([dist, d], dim=1)

        # get labels, contact with retained
        candidates = labels.view(1,-1).expand(len(testLabels), -1).to(device)
        trainLabels = candidates if trainLabels is None else torch.cat([trainLabels, candidates], dim=1)
        
        # get indexes, contact with retained
        # candidate_indexes = indexes.view(1,-1).expand(len(testLabels), -1).to(device)
        # trainIndexes = candidate_indexes if trainIndexes is None else torch.cat([trainIndexes, candidate_indexes], dim=1)
        
        # keep the top K distances and labels  
        #if batch_num % 10 == 0 or batch_num == (len(train_loader)-1):
        yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
        dist = torch.gather(dist, 1, yi)
        trainLabels = torch.gather(trainLabels, 1, yi)
        # trainIndexes = torch.gather(trainIndexes, 1, yi)
        
        trainFeatures, labels, indexes = None, None, None

    # generate weighted predictions
    batchSize = len(testLabels)
    retrieval = trainLabels
    retrieval_one_hot = torch.zeros(K, C).to('cpu')
    retrieval_one_hot.resize_(batchSize * K, C).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1).cpu(), 1)
    yd_transform = dist.clone().div_(sigma).exp_().cpu()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
    _, predictions = probs.sort(1, True)
    
    # Find which predictions match the target
    correct = predictions.eq(testLabels.view(-1,1).cpu())
    correct.shape
    
    total = correct.size(0)
    top1 = correct.narrow(1,0,1).sum().item() / total * 100
    top5 = correct.narrow(1,0,5).sum().item() / total * 100
    
    print(f"run_kNN accuracy: top1={top1}, top5={top5}")
    
    return top1, top5


def main():
    print('hello')


if __name__ == '__main__':
    main()
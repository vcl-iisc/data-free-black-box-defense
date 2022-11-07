import numpy as np
import torch
import torchvision

import setup


def loss(softmax, image, label):
    softmax = np.exp(softmax) / np.sum(np.exp(softmax))
    return np.sum(np.power(
        softmax -
        np.eye(10)[label],
        2
    ))


def optimize_to_grayscale(classifier, generator, batch_size=64, encoding_size=128):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        c = 0.
        x = 0
        specimens = np.random.uniform(-3.3, 3.3, size=(30, encoding_size))
        label = np.random.randint(10, size=(1, 1))
        while c < .9 and x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().to(setup.device))
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(setup.device)
                images = images * multipliers
                images = images.sum((1,), keepdim=True)

                softmaxes = classifier(images).detach().cpu()
            losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale=.5, size=(10, encoding_size)),
                specimens + np.random.normal(scale=.5, size=(10, encoding_size))
            ])
            c = softmaxes[indexes[0]][label]
        batch.append(image)
    return torch.cat(batch)  # , axis = 0)


def optimize_rescale(classifier, generator, batch_size=512):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 512
        c = 0.
        x = 0
        specimens = np.random.uniform(-3.3, 3.3, size=(20, encoding_size))
        label = np.random.randint(10, size=(1, 1))
        while c < .90 and x < 3:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float())
                images = torch.nn.functional.interpolate(images, size=224)
                # multipliers = [.2126, .7152, .0722]
                # multipliers = np.expand_dims(multipliers, 0)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.tile(multipliers, [1, 1, 32, 32])
                # multipliers = torch.Tensor(multipliers).to(device)
                # images = images * multipliers
                # images = images.sum(axis = 1, keepdims = True)

                softmaxes = classifier(images).detach().cpu()
            losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale=.5, size=(10, encoding_size)),
            ])
            c = softmaxes[indexes[0]][label]
        batch.append(image)
    return torch.cat(batch)  # , axis = 0)


def optimize(classifier, generator, batch_size=512, image_size=None, num_classes=10):
    batch = []
    n_random_samples = 30
    n_iter = batch_size
    encoding_size = 256

    if 'sngan' in str(type(generator)):
        encoding_size = 128

    if "ProgressiveGAN" in str(type(generator)):
        encoding_size = 512
    if image_size is not None:
        resize = torchvision.transforms.Resize(image_size)

    for i in range(n_iter):
        c = 0.
        x = 0

        specimens = np.random.uniform(-3.3, 3.3, size=(n_random_samples,
                                                       encoding_size))  # random noise for generator  #30 is number of sample to choose one image from
        label = np.random.randint(num_classes, size=(1, 1))  # label of image to be predicted

        while c < .9 and x < 300:  # while prediction probablity is <0.9  or number of iterations exceed 300  for current label
            x += 1
            encodings = specimens
            with torch.no_grad():

                if "ProgressiveGAN" in str(type(generator)):
                    images = generator.test(torch.tensor(
                        specimens
                    ).float().to(setup.device))

                else:
                    images = generator(torch.tensor(
                        specimens
                    ).float().to(setup.device))  # get images shape (30,3,32,32)

                if image_size is not None:
                    images = resize(images).float().to(setup.device)  # reshape if needed

                # multipliers = [.2126, .7152, .0722]
                # multipliers = np.expand_dims(multipliers, 0)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.tile(multipliers, [1, 1, 32, 32])
                # multipliers = torch.Tensor(multipliers).to(device)
                # images = images * multipliers
                # images = images.sum(axis = 1, keepdims = True)

                softmaxes = classifier(images).detach().cpu()  # get output of classifier for the images
            losses = [loss(np.array(s), i, label) for s, i in
                      zip(softmaxes, images)]  # compute mse loss for each image

            indexes = np.argsort(losses)  # get sorted indexes of losses
            image = images[indexes[0]: indexes[0] + 1]  # get the image with least loss
            specimens = specimens[indexes[:n_random_samples // 3]]  # get best 10 noise tensors with least loss
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale=.5, size=(n_random_samples // 3, encoding_size)),
                specimens + np.random.normal(scale=.5, size=(n_random_samples // 3, encoding_size))
            ])  # update noise
            c = softmaxes[indexes[0]].numpy()  # get softmax score for best image  10*1
            c = c[label]  # get probablity score for the given label
        batch.append(image)

    return torch.cat(batch)  # , axis = 0)


class Batch:
    def __init__(self):
        self.batch = []
        self.len = 0

    def size(self, dim):
        return self.len + 1

    def append(self, x):
        self.batch.append(x)
        self.len += x.size(0)


def optimize_new_2(classifier, generator, batch_size=512, image_size=None, num_classes=10):
    # batch = torch.randn((1,3,32,32)).float().to(setup.device)     #one extra dummy vector is defined in batch to avoid None value exception
    batch = Batch()
    batch_size += 1  # increase batch size by 1 to take care of first dummy element in  batch
    mse_loss = torch.nn.MSELoss(reduction="none")
    sample_per_class = batch_size // num_classes + 1  # 512,10 = 51+1 = 52  # each class label is equally represented in batch

    n_random_samples = 30
    n_sample_step = 32  # number of images to create at one step
    encoding_size = 256

    if 'sngan' in str(type(generator)):
        encoding_size = 128

    if "ProgressiveGAN" in str(type(generator)):
        encoding_size = 512

    resize = None
    if image_size is not None:
        resize = torchvision.transforms.Resize(image_size)

    random_labels = torch.randperm(num_classes)  # randomly select labels

    x = torch.zeros(size=(n_sample_step,),
                    device=setup.device)  # for each image we keep track of number of iterations.
    threshold = torch.zeros(size=(n_sample_step,),
                            device=setup.device)  # shape=(32*1)                                                                #After every 100 iterations we decrease threshold to avoid being stuck in loop

    label_tensor = torch.full((n_sample_step, n_random_samples), 0, device=setup.device)  # 32*30 labels with same class

    specimens_bucket = torch.zeros((n_sample_step * 32, n_random_samples, encoding_size), device=setup.device).uniform_(
        -3, 3)
    s_index = 0

    def get_speciemens(size):
        nonlocal s_index
        nonlocal specimens_bucket

        if s_index + size > specimens_bucket.size(0):
            specimens_bucket = torch.zeros((n_sample_step * 32, n_random_samples, encoding_size),
                                           device=setup.device).uniform_(-3, 3)
            s_index = 0

        specimens = specimens_bucket[s_index:s_index + size]
        s_index += size
        return specimens

    s1 = torch.zeros((n_sample_step, n_random_samples // 3, encoding_size), device=setup.device)
    s2 = torch.zeros((n_sample_step, n_random_samples // 3, encoding_size), device=setup.device)

    while batch.size(0) <= batch_size:
        c = None

        x *= 0  # reset all entries in x to zero

        specimens = get_speciemens(n_sample_step)

        current_label_value = random_labels[batch.size(
            0) // sample_per_class]  # get current label. we sample #sample_per_class number of images for a label and then move to next label
        current_label_index = batch.size(0) // sample_per_class  # inde

        label_tensor.fill_(current_label_value)
        label = torch.nn.functional.one_hot(label_tensor, num_classes=num_classes)  # convert to one hot

        threshold.fill_(0.9)

        while batch.size(
                0) // sample_per_class == current_label_index:  # while sufficient images are not added for current label
            x += 1

            y = torch.where(x % 100 != 0, 0, 0.1)  # reduce threshold for images that are stuck for long time
            threshold -= y

            with torch.no_grad():

                specimens = specimens.view(-1, encoding_size)  # convert shape to (32*30, encoding size)
                if "ProgressiveGAN" in str(type(generator)):
                    noise, _ = generator.buildNoiseData(n_sample_step * n_random_samples)
                    images = generator.test(noise).to(setup.device)
                else:
                    images = generator(specimens)  # get images  shape is (32*30, 3 , 32, 32)

                # if resize:
                #   images = resize(images)

                specimens = specimens.view(n_sample_step, n_random_samples,
                                           encoding_size)  # convert back to original shape  shape=(32,30,128)

                output = classifier(images)  # get classifier output for images  shape = (32*30, 10)

                output = output.view(n_sample_step, n_random_samples, num_classes)  # reshape  shape=(32,30,10)

            softmaxes = torch.nn.functional.softmax(output, dim=2)  # get softmax output
            losses = torch.sum(mse_loss(softmaxes, label), dim=2)  # calculate loss for each image shape =(32*30,)

            images = images.view(n_sample_step, n_random_samples, 3, image_size,
                                 image_size)  # reshape images shape = (32,30,3,32,32)

            indexes = torch.argsort(losses, dim=1)  # sort losses at dimension-1

            # select index of best loss
            first_index = indexes[:, 0:1].view(n_sample_step, 1, 1, 1, 1).expand(-1, -1, 3, image_size, image_size)

            first_ten_index = indexes[:, 0:n_random_samples // 3].view(n_sample_step, n_random_samples // 3, 1).expand(
                -1, -1, encoding_size)

            image = torch.gather(images, 1, first_index).squeeze(
                1)  # select best out of 30 samples for each of 12 images  shape=(12,3,32,32)

            best_ten_specimens = torch.gather(specimens, 1,
                                              first_ten_index)  # select top 10 specimens for each of 12 images  shape=(12,10,encoding_size)

            specimens = torch.cat([
                best_ten_specimens,
                best_ten_specimens + s1.normal_(0.5, 0.5),
                best_ten_specimens + s2.normal_(0.5, 0.5)
            ], dim=1)  # concatenate

            first_index_softmax = indexes[:, 0:1].view(n_sample_step, 1, 1).expand(-1, -1, num_classes)

            c = torch.gather(output, 1, first_index_softmax).squeeze(
                1)  # for each of 32 images, selct softmax output of best out of 30 samples shape=(12,10)
            c = c[:, current_label_value]  # for each of 32 images select softmax score for the label

            index_list = (c > threshold).int().nonzero().squeeze(
                1)  # find for which images softmax score is more than 0.9  shape =(-1,30,128)

            images = torch.index_select(image, 0, index_list)  # add selected images to batch

            index_list = index_list.view(-1, 1, 1).expand(-1, n_random_samples,
                                                          128)  # change dimension of index_list to match with specimens shape

            new_specimens = get_speciemens(index_list.size(0))

            specimens.scatter_(dim=0, index=index_list,
                               src=new_specimens)  # update specimens for selected images

            threshold = torch.where(c > threshold, 0.9,
                                    threshold)  # for selected images reset threshold back to 0.9 shape
            x = torch.where(c > threshold, 0, x)

            batch.append(images)

    batch = torch.cat(batch.batch)

    return torch.index_select(batch, 0, torch.randperm(len(batch), device=setup.device))


def discrepancy_loss(teacher_predictions, student_predictions):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return -np.sum(np.square(teacher_softmax - student_softmax))


def optimize_discrepancies(teacher, student, generator, batch_size=16):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size=(30, encoding_size))
        x = 0
        while x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().to(setup.device))
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(setup.device)
                images = images * multipliers
                images = images.sum(axis=1, keepdims=True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                discrepancy_loss(np.array(s), np.array(i))
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale=.5, size=(10, encoding_size)),
                specimens + np.random.normal(scale=.5, size=(10, encoding_size))
            ])
        batch.append(image)
    return torch.cat(batch, axis=0)


def discrepancy_loss_kl(teacher_predictions, student_predictions):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return -np.sum(teacher_softmax * np.log(teacher_softmax) / np.log(student_softmax))


def optimize_discrepancies_kl(teacher, student, generator, batch_size=64):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size=(30, encoding_size))
        x = 0
        while x < 50:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().to(setup.device))
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(setup.device)
                images = images * multipliers
                images = images.sum(axis=1, keepdims=True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                discrepancy_loss_kl(np.array(s), np.array(i))
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale=.5, size=(10, encoding_size)),
                specimens + np.random.normal(scale=.5, size=(10, encoding_size))
            ])
        batch.append(image)
        print('image')
    return torch.cat(batch, axis=0)


def loss(softmax, image, label):
    softmax = np.exp(softmax) / np.sum(np.exp(softmax))
    dim = softmax.shape[0]
    return np.sum(np.power(
        softmax -
        np.eye(dim)[label],
        2
    ))


def optimize_discrepancies_(teacher, student, generator, batch_size=64):
    encoding_size = 128
    batch_size = 16

    with torch.no_grad():
        specimens = torch.tensor(
            np.random.uniform(-3.3, 3.3, size=(batch_size, 30, encoding_size))
        ).float().to(setup.device)

        for _ in range(10):
            images = generator(specimens.view(-1, encoding_size))
            multipliers = [.2126, .7152, .0722]
            multipliers = np.expand_dims(multipliers, 0)
            multipliers = np.expand_dims(multipliers, -1)
            multipliers = np.expand_dims(multipliers, -1)
            multipliers = np.tile(multipliers, [1, 1, 32, 32])
            multipliers = torch.Tensor(multipliers).to(setup.device)
            images = images * multipliers
            images = images.sum(axis=1, keepdims=True)
            teacher_predictions = torch.softmax(
                teacher(images), axis=-1
            ).detach().cpu()
            student_predictions = torch.softmax(
                student(images), axis=-1
            ).detach().cpu()

            losses = -1. * torch.pow(
                teacher_predictions - student_predictions, 2
            ).sum(-1).view(batch_size, 30)
            indexes = torch.argsort(losses) < 10
            specimens = specimens[indexes].view(batch_size, 10, encoding_size)
            specimens = torch.cat((
                specimens,
                specimens + torch.randn(batch_size, 10, encoding_size).to(setup.device),
                specimens + torch.randn(batch_size, 10, encoding_size).to(setup.device),
            ), axis=1)

        images = generator(specimens.view(-1, encoding_size))
        images = generator(specimens.view(-1, encoding_size))
        multipliers = [.2126, .7152, .0722]
        multipliers = np.expand_dims(multipliers, 0)
        multipliers = np.expand_dims(multipliers, -1)
        multipliers = np.expand_dims(multipliers, -1)
        multipliers = np.tile(multipliers, [1, 1, 32, 32])
        multipliers = torch.Tensor(multipliers).to(setup.device)
        images = images * multipliers
        images = images.sum(axis=1, keepdims=True)

        teacher_predictions = torch.softmax(
            teacher(images), axis=-1
        ).detach().cpu()
        student_predictions = torch.softmax(
            student(images), axis=-1
        ).detach().cpu()

        losses = -1. * torch.pow(
            teacher_predictions - student_predictions, 2
        ).sum(-1).view(batch_size, 30)

        indexes = torch.argsort(losses) < 1
        images = images.view(batch_size, 30, 1, 32, 32)[indexes]
    return images


def curriculum_loss(teacher_predictions, student_predictions, label, weight):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return (
            np.sum(np.square(teacher_softmax - label)) -
            weight * np.sum(np.square(teacher_softmax - student_softmax))
    )


def optimize_curriculum(teacher, student, generator, epoch, batch_size=16):
    batch = []
    weights = [0.] * 4 + list(np.linspace(0, 1., 46)) + [1.] * 200

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size=(30, encoding_size))
        x = 0
        label = np.eye(10)[np.random.randint(10)]
        while x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().cuda())
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(setup.device)
                images = images * multipliers
                images = images.sum(axis=1, keepdims=True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                curriculum_loss(np.array(s), np.array(i), label, weights[epoch])
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale=.5, size=(10, encoding_size)),
                specimens + np.random.normal(scale=.5, size=(10, encoding_size))
            ])
        batch.append(image)
    return torch.cat(batch, axis=0)

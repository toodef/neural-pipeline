Create TrainConfig
==================

Now let's define ``TrainConfig`` that will contains training hyperparameters.

In this tutorial we use predefined stages ``TrainStage`` and ``ValidationStage``. ``TrainStage`` iterate by ``DataProducer`` and learn model in ``train()`` mode.
Respectively ``ValidatioStage`` do same but in ``eval()`` mode.

.. code:: python

    from piepline import TrainConfig, TrainStage, ValidationStage

    # define train stages
    train_stages = [TrainStage(train_dataset), ValidationStage(validation_dataset)]

    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)

    # define TrainConfig
    train_config = TrainConfig(train_stages, loss, optimizer)
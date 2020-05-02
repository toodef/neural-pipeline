Continue training
=================

If we need to do some more training epochs but doesn't have previously defined objects we need to do this:

.. code:: python

    # define again all from previous steps
    # ...

    # define FileStructureManager with parameter is_continue=True
    fsm = FileStructManager(base_dir='data', is_continue=True)

    # create trainer
    trainer = Trainer(model, train_config, fsm, torch.device('cuda:0'))

    # specify training epochs number
    trainer.set_epoch_num(50)

    # add TensorboardMonitor with parameter is_continue=True
    trainer.monitor_hub.add_monitor(TensorboardMonitor(fsm, is_continue=True))

    # set Trainer to resume mode and run training
    trainer.resume(from_best_checkpoint=False).train()


Parameter ``from_best_checkpoint=False`` tell Trainer, that it need continue from last checkpoint.
PiePline can save best checkpoints by specified rule. For more information about it read about `enable_lr_decaying <https://piepline.readthedocs.io/en/master/api/train.html#piepline.train.Trainer.enable_best_states_saving>`_ method of `Trainer`.

Don't worry about incorrect training history displaying. If history also exists - monitors just add new data to it.
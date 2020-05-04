DVC
===

Portrait segmentation network.

This based on PyTorch, PiePline and high-level pipeline build by [DVC](dvc.org).

Creation repo tutorial (explain, that code also exists):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This steps also done and results contains in repo. For reproduce this step make:
```
dvc destroy
git commit -m 'deinit DVC'
```
###Clone repo

1) add PixArt dataset as submodule
```
git submodule add http://172.26.40.23:3000/datasets/pixart.git datasets/
```
2) load all from submodule
```
git submodule update --init
```

###Build DVC pipeline:

1) initialize DVC
```
dvc init
git commit -m 'add DVC'
```
2) Setup pipeline
```
dvc run -d train.py -M data/monitors/metrics_log/metrics.json -o data/checkpoints/last/last_checkpoint.zip --no-exec python train.py
dvc run -d predict.py -d data/checkpoints/last/last_checkpoint.zip -o result --no-exec python predict.py
```
3) Run pipeline
```
dvc repro result.dvc
```
4) Last steps

After pipeline execution end, we get `metrics.json` file with metrics values and pipeline modified steps files. Let's add it to git history
```
git add data/checkpoints/last/.gitignore last_checkpoint.zip.dvc result.dvc metrics.json -f
```

###Run another experiment
We add hard negative mining to our training process. So we need to run new experiment and then compare it with existing

1) Create new branch

```
git checkout -b hnm
dvc checkout
```

2) Repeat all steps from previous section

3) Compare metrics

```
dvc metrics show -a
```

Output will look like that:

```
hnm:
    metrics.json: {"train": {"jaccard": 0.8874640464782715, "dice": 0.9423233270645142, "loss": 0.7522647976875305}, "validation": {"jaccard": 0.8573445081710815, "dice": 0.9246319532394409, "loss": 0.7623925805091858}}
master:
    metrics.json: {"train": {"jaccard": 0.8774164915084839, "dice": 0.9357065558433533, "loss": 0.7595105767250061}, "validation": {"jaccard": 0.8574965596199036, "dice": 0.927370011806488, "loss": 0.7602806687355042}}
```


###Show DVC pipeline:
```
dvc pipeline show --ascii result.dvc
```
U may see this output:
```
+-------------------------+
| last_checkpoint.zip.dvc |
+-------------------------+
              *
              *
              *
      +------------+
      | result.dvc |
      +------------+
```

## Reproduce results:
Call `dvc repro` will run pipeline. But we need define last step of pipeline. So as a parameter we pass last pipeline step file name:
```
dvc repro result.dvc
```

After pipeline stop executing, you can see metrics (`-a` - show metrics from all branches):
```
dvc metrics show -a
```
# dump

Default directory for file storage.
All downloads, input data, model outputs, plots, tables,
and log files are stored in the `dump` directory by default.

Running the script `produce_all_results.py` will produce
12.5 GB of permanent files and over 300 GB of temporary files,
all written into `dump` by default.
If you would like to store these files in a different location,
activate the `eval14c` environment and run the following command:

```
python -m config -set-dump "/your/new/path/to/dump"
```

You can check the absolute path of the current file storage location with

```
python -m config -get-dump
```
# pyglmnet documentation

See full documentation page [here](http://pavanramkumar.github.io/pyglmnet/).

We use `sphinx` to generate documentation page.
You can install dependencies for documentation page by running

```python
$ pip install -r requirements.txt
```

To build documentation page, run `make html`. All static files will be built in
`_build/html/` where you can run simple server to see documentation locally.
Here is example on how to run at port 8000 (`http://localhost:8000`).

```bash
$ python -m SimpleHTTPServer 8000 # for Python 2
$ python -m http.server 8000 # for Python 3
```

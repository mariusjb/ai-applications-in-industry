```python
%%time

!pip install -r requirements.txt
```

    Requirement already satisfied: jupyterlab in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from -r requirements.txt (line 1)) (4.3.6)
    Requirement already satisfied: pandas in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from -r requirements.txt (line 2)) (2.2.3)
    Requirement already satisfied: numpy in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from -r requirements.txt (line 3)) (2.0.2)
    Requirement already satisfied: scikit-learn in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from -r requirements.txt (line 4)) (1.6.1)
    Requirement already satisfied: xgboost in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from -r requirements.txt (line 5)) (3.0.0)
    Requirement already satisfied: plotly in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from -r requirements.txt (line 6)) (6.0.0)
    Requirement already satisfied: async-lru>=1.0.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (2.0.5)
    Requirement already satisfied: httpx>=0.25.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (0.28.1)
    Requirement already satisfied: ipykernel>=6.5.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (6.29.5)
    Requirement already satisfied: jinja2>=3.0.3 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (3.1.5)
    Requirement already satisfied: jupyter-core in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (5.7.2)
    Requirement already satisfied: jupyter-lsp>=2.0.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (2.2.5)
    Requirement already satisfied: jupyter-server<3,>=2.4.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (2.15.0)
    Requirement already satisfied: jupyterlab-server<3,>=2.27.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (2.27.3)
    Requirement already satisfied: notebook-shim>=0.2 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (0.2.4)
    Requirement already satisfied: packaging in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (24.2)
    Requirement already satisfied: setuptools>=40.8.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (75.8.0)
    Requirement already satisfied: tomli>=1.2.2 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (2.2.1)
    Requirement already satisfied: tornado>=6.2.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (6.4.2)
    Requirement already satisfied: traitlets in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab->-r requirements.txt (line 1)) (5.14.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from pandas->-r requirements.txt (line 2)) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from pandas->-r requirements.txt (line 2)) (2025.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from pandas->-r requirements.txt (line 2)) (2025.1)
    Requirement already satisfied: scipy>=1.6.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from scikit-learn->-r requirements.txt (line 4)) (1.15.2)
    Requirement already satisfied: joblib>=1.2.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from scikit-learn->-r requirements.txt (line 4)) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from scikit-learn->-r requirements.txt (line 4)) (3.5.0)
    Requirement already satisfied: narwhals>=1.15.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from plotly->-r requirements.txt (line 6)) (1.29.0)
    Requirement already satisfied: typing_extensions>=4.0.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from async-lru>=1.0.0->jupyterlab->-r requirements.txt (line 1)) (4.12.2)
    Requirement already satisfied: anyio in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from httpx>=0.25.0->jupyterlab->-r requirements.txt (line 1)) (4.9.0)
    Requirement already satisfied: certifi in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from httpx>=0.25.0->jupyterlab->-r requirements.txt (line 1)) (2025.1.31)
    Requirement already satisfied: httpcore==1.* in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from httpx>=0.25.0->jupyterlab->-r requirements.txt (line 1)) (1.0.7)
    Requirement already satisfied: idna in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from httpx>=0.25.0->jupyterlab->-r requirements.txt (line 1)) (3.10)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from httpcore==1.*->httpx>=0.25.0->jupyterlab->-r requirements.txt (line 1)) (0.14.0)
    Requirement already satisfied: appnope in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (0.1.4)
    Requirement already satisfied: comm>=0.1.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (0.2.2)
    Requirement already satisfied: debugpy>=1.6.5 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (1.8.11)
    Requirement already satisfied: ipython>=7.23.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (8.32.0)
    Requirement already satisfied: jupyter-client>=6.1.12 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (8.6.3)
    Requirement already satisfied: matplotlib-inline>=0.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (0.1.7)
    Requirement already satisfied: nest-asyncio in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (1.6.0)
    Requirement already satisfied: psutil in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (5.9.0)
    Requirement already satisfied: pyzmq>=24 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (26.2.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jinja2>=3.0.3->jupyterlab->-r requirements.txt (line 1)) (3.0.2)
    Requirement already satisfied: platformdirs>=2.5 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-core->jupyterlab->-r requirements.txt (line 1)) (4.3.6)
    Requirement already satisfied: argon2-cffi>=21.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (23.1.0)
    Requirement already satisfied: jupyter-events>=0.11.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.12.0)
    Requirement already satisfied: jupyter-server-terminals>=0.4.4 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.5.3)
    Requirement already satisfied: nbconvert>=6.4.4 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (7.16.6)
    Requirement already satisfied: nbformat>=5.3.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (5.10.4)
    Requirement already satisfied: overrides>=5.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (7.7.0)
    Requirement already satisfied: prometheus-client>=0.9 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.21.1)
    Requirement already satisfied: send2trash>=1.8.2 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (1.8.3)
    Requirement already satisfied: terminado>=0.8.3 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.18.1)
    Requirement already satisfied: websocket-client>=1.7 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (1.8.0)
    Requirement already satisfied: babel>=2.10 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (2.17.0)
    Requirement already satisfied: json5>=0.9.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (0.10.0)
    Requirement already satisfied: jsonschema>=4.18.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (4.23.0)
    Requirement already satisfied: requests>=2.31 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (2.32.3)
    Requirement already satisfied: six>=1.5 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas->-r requirements.txt (line 2)) (1.17.0)
    Requirement already satisfied: exceptiongroup>=1.0.2 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from anyio->httpx>=0.25.0->jupyterlab->-r requirements.txt (line 1)) (1.2.2)
    Requirement already satisfied: sniffio>=1.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from anyio->httpx>=0.25.0->jupyterlab->-r requirements.txt (line 1)) (1.3.1)
    Requirement already satisfied: argon2-cffi-bindings in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (21.2.0)
    Requirement already satisfied: decorator in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (5.2.1)
    Requirement already satisfied: jedi>=0.16 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (0.19.2)
    Requirement already satisfied: pexpect>4.3 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (4.9.0)
    Requirement already satisfied: prompt_toolkit<3.1.0,>=3.0.41 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (3.0.50)
    Requirement already satisfied: pygments>=2.4.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (2.19.1)
    Requirement already satisfied: stack_data in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (0.6.3)
    Requirement already satisfied: attrs>=22.2.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (25.1.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (2024.10.1)
    Requirement already satisfied: referencing>=0.28.4 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (0.36.2)
    Requirement already satisfied: rpds-py>=0.7.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (0.23.1)
    Requirement already satisfied: python-json-logger>=2.0.4 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (3.3.0)
    Requirement already satisfied: pyyaml>=5.3 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (6.0.2)
    Requirement already satisfied: rfc3339-validator in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.1.4)
    Requirement already satisfied: rfc3986-validator>=0.1.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.1.1)
    Requirement already satisfied: beautifulsoup4 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (4.13.3)
    Requirement already satisfied: bleach!=5.0.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from bleach[css]!=5.0.0->nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (6.2.0)
    Requirement already satisfied: defusedxml in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.7.1)
    Requirement already satisfied: jupyterlab-pygments in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.3.0)
    Requirement already satisfied: mistune<4,>=2.0.3 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (3.1.3)
    Requirement already satisfied: nbclient>=0.5.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.10.2)
    Requirement already satisfied: pandocfilters>=1.4.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (1.5.1)
    Requirement already satisfied: fastjsonschema>=2.15 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from nbformat>=5.3.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (2.21.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from requests>=2.31->jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (3.4.1)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from requests>=2.31->jupyterlab-server<3,>=2.27.1->jupyterlab->-r requirements.txt (line 1)) (2.3.0)
    Requirement already satisfied: ptyprocess in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from terminado>=0.8.3->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.7.0)
    Requirement already satisfied: webencodings in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from bleach!=5.0.0->bleach[css]!=5.0.0->nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (0.5.1)
    Requirement already satisfied: tinycss2<1.5,>=1.1.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from bleach[css]!=5.0.0->nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (1.4.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jedi>=0.16->ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (0.8.4)
    Requirement already satisfied: fqdn in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (1.5.1)
    Requirement already satisfied: isoduration in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (20.11.0)
    Requirement already satisfied: jsonpointer>1.13 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (3.0.0)
    Requirement already satisfied: uri-template in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: webcolors>=24.6.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (24.11.1)
    Requirement already satisfied: wcwidth in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from prompt_toolkit<3.1.0,>=3.0.41->ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (0.2.13)
    Requirement already satisfied: cffi>=1.0.1 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (1.17.1)
    Requirement already satisfied: soupsieve>1.2 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from beautifulsoup4->nbconvert>=6.4.4->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (2.6)
    Requirement already satisfied: executing>=1.2.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from stack_data->ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (2.1.0)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from stack_data->ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (3.0.0)
    Requirement already satisfied: pure_eval in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from stack_data->ipython>=7.23.1->ipykernel>=6.5.0->jupyterlab->-r requirements.txt (line 1)) (0.2.3)
    Requirement already satisfied: pycparser in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (2.22)
    Requirement already satisfied: arrow>=0.15.0 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: types-python-dateutil>=2.8.10 in /opt/miniconda3/envs/ai-applications-in-industry/lib/python3.10/site-packages (from arrow>=0.15.0->isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.11.0->jupyter-server<3,>=2.4.0->jupyterlab->-r requirements.txt (line 1)) (2.9.0.20241206)
    CPU times: user 7.9 ms, sys: 14.8 ms, total: 22.7 ms
    Wall time: 1.01 s


### Data Pipeline
Load data from `data/PJME_hourly` and engineer features to enrich the temporal context.

**Note: Lag features and rolling features can be enabled / disabled to provide more / less contextual information.**


```python
%%time

from utils.data_utils import (
    load_and_split_data,
    add_time_features,
    add_lag_features,
    add_rolling_features,
)

train, val, test = load_and_split_data(path="data/PJME_hourly.csv")
dfs = {"train": train, "val": val, "test": test}

for name, df in dfs.items():
    dfs[name] = (
        df.pipe(add_time_features)
        # .pipe(add_lag_features)  # Uncomment to add lag features
        .pipe(add_rolling_features)  # Uncomment to add rolling features
        .pipe(lambda x: x.dropna())
    )
    print(f"\n{name}: {dfs[name].shape}")
    print(dfs[name].head(2))

train, val, test = dfs.values()
```

    
    train: (101732, 10)
                  Datetime  PJME_MW  hour  dayofweek  month  day  year  \
    24 2002-01-02 01:00:00  28121.0     1          2      1    2  2002   
    25 2002-01-02 02:00:00  27437.0     2          2      1    2  2002   
    
        is_weekend  rolling_mean_24  rolling_std_24  
    24           0     31017.500000     2423.666231  
    25           0     30922.833333     2492.512817  
    
    val: (21781, 10)
                      Datetime  PJME_MW  hour  dayofweek  month  day  year  \
    101780 2013-08-12 21:00:00  42567.0    21          0      8   12  2013   
    101781 2013-08-12 22:00:00  40735.0    22          0      8   12  2013   
    
            is_weekend  rolling_mean_24  rolling_std_24  
    101780           0     36604.208333     6865.057389  
    101781           0     36783.458333     6965.732334  
    
    test: (21781, 10)
                      Datetime  PJME_MW  hour  dayofweek  month  day  year  \
    123585 2016-02-07 11:00:00  30911.0    11          6      2    7  2016   
    123586 2016-02-07 12:00:00  30504.0    12          6      2    7  2016   
    
            is_weekend  rolling_mean_24  rolling_std_24  
    123585           1     30452.958333     1822.581363  
    123586           1     30400.375000     1788.688658  
    CPU times: user 89.1 ms, sys: 18.8 ms, total: 108 ms
    Wall time: 111 ms


### XGBoost
XGBoost is a strong baseline for time series forecasting tasks, having demonstrated competitive performance in various studies. In this experiment, it serves as a benchmark to evaluate the effectiveness of selected deep learning models in forecasting energy consumption.


```python
%%time

import xgboost as xgb
from utils.data_utils import prepare_datasets
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from utils.visualization_utils import plot_actual_vs_pred

X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(train, val, test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "max_depth": 6,
    "eta": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}
num_round = 5000
evallist = [(dtrain, "train"), (dval, "eval")]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=evallist,
    # early_stopping_rounds=20,
    verbose_eval=250,
)

xgb_y_pred = model.predict(dtest)
xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
xgb_rmse = root_mean_squared_error(y_test, xgb_y_pred)

print(f"[XGBoost] Test MAE: {xgb_mae:.2f}")
print(f"[XGBoost] Test RMSE: {xgb_rmse:.2f}")

plot_actual_vs_pred(y_test, xgb_y_pred, title="XGBoost")
```

    [0]	train-rmse:6335.73867	eval-rmse:6095.49377
    [250]	train-rmse:1117.30170	eval-rmse:1377.35045
    [500]	train-rmse:984.24516	eval-rmse:1333.09550
    [750]	train-rmse:904.82500	eval-rmse:1331.21435
    [1000]	train-rmse:846.67837	eval-rmse:1331.46384
    [1250]	train-rmse:801.20101	eval-rmse:1333.90937
    [1500]	train-rmse:760.17604	eval-rmse:1343.52932
    [1750]	train-rmse:726.48535	eval-rmse:1348.01226
    [2000]	train-rmse:695.19607	eval-rmse:1357.10617
    [2250]	train-rmse:668.19813	eval-rmse:1363.46727
    [2500]	train-rmse:643.41388	eval-rmse:1369.88500
    [2750]	train-rmse:621.06849	eval-rmse:1377.56376
    [3000]	train-rmse:600.66287	eval-rmse:1384.05724
    [3250]	train-rmse:582.01856	eval-rmse:1389.38264
    [3500]	train-rmse:564.26083	eval-rmse:1396.50349
    [3750]	train-rmse:547.65349	eval-rmse:1401.24610
    [4000]	train-rmse:532.08689	eval-rmse:1408.21971
    [4250]	train-rmse:517.36989	eval-rmse:1414.01770
    [4500]	train-rmse:503.48619	eval-rmse:1418.63173
    [4750]	train-rmse:490.65735	eval-rmse:1422.93649
    [4999]	train-rmse:478.69310	eval-rmse:1426.76520
    [XGBoost] Test MAE: 1148.90
    [XGBoost] Test RMSE: 1581.88




    CPU times: user 47.5 s, sys: 5.6 s, total: 53.1 s
    Wall time: 7.06 s


### Data Pipeline - P2
We prepare the data for one-step-ahead forecasting with a PyTorch model by following these steps: scaling the data, creating input sequences, reating a PyTorch Dataset, and batching with DataLoader.


```python
%%time

import numpy as np
from torch.utils.data import DataLoader
from utils.data_utils import scale, create_sequences
from utils.dataset import Dataset_ECF

# Takes some time; keep your eyes on memory usage

train_scaled, val_scaled, test_scaled, feature_scaler, target_scaler = scale(
    train, val, test, ignore_cols=["is_weekend"]
)

X_train, y_train = create_sequences(train_scaled, window_size=24)
X_val, y_val = create_sequences(val_scaled, window_size=24)
X_test, y_test = create_sequences(test_scaled, window_size=24)

# Shuffle the training data before creating DataLoader
perm = np.random.permutation(len(X_train))
X_train_shuffled = X_train[perm]
y_train_shuffled = y_train[perm]

train_ds = Dataset_ECF(X_train_shuffled, y_train_shuffled)
val_ds = Dataset_ECF(X_val, y_val)
test_ds = Dataset_ECF(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, drop_last=True)  # shuffle=False as we have already shuffled
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, drop_last=True)

print(f"Train Dataloader: {len(train_dl)}")
print(f"Val Dataloader: {len(val_dl)}")
print(f"Test Dataloader: {len(test_dl)}")

```

    100%|██████████| 101708/101708 [01:22<00:00, 1236.09it/s]
    100%|██████████| 21757/21757 [00:04<00:00, 5017.51it/s]
    100%|██████████| 21757/21757 [00:04<00:00, 5025.78it/s]

    Train Dataloader: 794
    Val Dataloader: 169
    Test Dataloader: 169
    CPU times: user 1min 6s, sys: 25.4 s, total: 1min 32s
    Wall time: 1min 31s


    


### Long Short-Term Memory
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to model sequential data. It can capture temporal dependencies and seasonal patterns in time series by maintaining a memory of past observations. It's the first PyTorch model in this experiment.


```python
device = "mps"  # Defined once; change to "cpu" or - if available - "cuda"
```


```python
%%time

import torch
from models.lstm import LSTM_ECF
from utils.training_utils import set_seed, train, evaluate
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils.visualization_utils import plot_actual_vs_pred

set_seed(42)

lstm_model = LSTM_ECF(input_size=X_train.shape[2])

lstm_model = train(
    lstm_model,
    train_loader=train_dl,
    val_loader=val_dl,
    optimizer=torch.optim.Adam(lstm_model.parameters(), lr=1e-3),
    loss_fn=torch.nn.MSELoss(),
    num_epochs=30,
    patience=3,
    device=device,
)

lstm_y_true, lstm_y_pred = evaluate(lstm_model, test_dl, device)
lstm_y_true = target_scaler.inverse_transform(lstm_y_true.reshape(-1, 1)).flatten()
lstm_y_pred = target_scaler.inverse_transform(lstm_y_pred.reshape(-1, 1)).flatten()

lstm_mae = mean_absolute_error(lstm_y_true, lstm_y_pred)
lstm_rmse = root_mean_squared_error(lstm_y_true, lstm_y_pred)

print(f"[LSTM] Test MAE: {lstm_mae:.2f}")
print(f"[LSTM] Test RMSE: {lstm_rmse:.2f}")

plot_actual_vs_pred(lstm_y_true, lstm_y_pred, title="LSTM")
```

    Epoch 01: 100%|██████████| 794/794 [00:11<00:00, 68.34it/s]


    Epoch 01 | Train Loss: 0.0800 | Val Loss: 0.0608


    Epoch 02: 100%|██████████| 794/794 [00:10<00:00, 73.18it/s]


    Epoch 02 | Train Loss: 0.0430 | Val Loss: 0.0481


    Epoch 03: 100%|██████████| 794/794 [00:10<00:00, 73.22it/s]


    Epoch 03 | Train Loss: 0.0358 | Val Loss: 0.0446


    Epoch 04: 100%|██████████| 794/794 [00:10<00:00, 74.65it/s]


    Epoch 04 | Train Loss: 0.0314 | Val Loss: 0.0433


    Epoch 05: 100%|██████████| 794/794 [00:10<00:00, 72.51it/s]


    Epoch 05 | Train Loss: 0.0283 | Val Loss: 0.0418


    Epoch 06: 100%|██████████| 794/794 [00:10<00:00, 72.57it/s]


    Epoch 06 | Train Loss: 0.0259 | Val Loss: 0.0395


    Epoch 07: 100%|██████████| 794/794 [00:11<00:00, 70.73it/s]


    Epoch 07 | Train Loss: 0.0240 | Val Loss: 0.0375


    Epoch 08: 100%|██████████| 794/794 [00:11<00:00, 70.39it/s]


    Epoch 08 | Train Loss: 0.0224 | Val Loss: 0.0362


    Epoch 09: 100%|██████████| 794/794 [00:10<00:00, 73.16it/s]


    Epoch 09 | Train Loss: 0.0214 | Val Loss: 0.0350


    Epoch 10: 100%|██████████| 794/794 [00:11<00:00, 71.99it/s]


    Epoch 10 | Train Loss: 0.0201 | Val Loss: 0.0352


    Epoch 11: 100%|██████████| 794/794 [00:10<00:00, 74.17it/s]


    Epoch 11 | Train Loss: 0.0191 | Val Loss: 0.0377


    Epoch 12: 100%|██████████| 794/794 [00:11<00:00, 69.65it/s]


    Epoch 12 | Train Loss: 0.0182 | Val Loss: 0.0353
    Early stopping triggered at epoch 12
    [LSTM] Test MAE: 1058.46
    [LSTM] Test RMSE: 1421.27




    CPU times: user 1min 11s, sys: 8.83 s, total: 1min 20s
    Wall time: 2min 27s


**LSTM improves forecasts compared to XGBoost.**

### Temporal Convolutional Networks
Temporal Convolutional Networks (TCNs) are an alternative to recurrent models for handling sequential data. Unlike LSTMs, which process inputs step by step, TCNs use causal and dilated convolutions to capture temporal dependencies, enabling them to process entire sequences in parallel. TCNs are particularly effective at learning long-term patterns in time series data.


```python
%%time

import torch
from models.tcn import TCN_ECF
from utils.training_utils import set_seed, train, evaluate
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils.visualization_utils import plot_actual_vs_pred

set_seed(42)

tcn_model = TCN_ECF(
    input_size=X_train.shape[2], num_channels=[64] * 4, kernel_size=5, dropout=0.3
)

tcn_model = train(
    tcn_model,
    train_loader=train_dl,
    val_loader=val_dl,
    optimizer=torch.optim.Adam(tcn_model.parameters(), lr=1e-3),
    loss_fn=torch.nn.MSELoss(),
    num_epochs=30,
    patience=3,
    device=device,
)

tcn_y_true, tcn_y_pred = evaluate(tcn_model, test_dl, device)
tcn_y_true = target_scaler.inverse_transform(tcn_y_true.reshape(-1, 1)).flatten()
tcn_y_pred = target_scaler.inverse_transform(tcn_y_pred.reshape(-1, 1)).flatten()

tcn_mae = mean_absolute_error(tcn_y_true, tcn_y_pred)
tcn_rmse = root_mean_squared_error(tcn_y_true, tcn_y_pred)

print(f"[TCN] Test MAE: {tcn_mae:.2f}")
print(f"[TCN] Test RMSE: {tcn_rmse:.2f}")

plot_actual_vs_pred(tcn_y_true, tcn_y_pred, title="TCN")
```

    Epoch 01: 100%|██████████| 794/794 [00:12<00:00, 62.16it/s]


    Epoch 01 | Train Loss: 0.0830 | Val Loss: 0.0513


    Epoch 02: 100%|██████████| 794/794 [00:12<00:00, 65.39it/s]


    Epoch 02 | Train Loss: 0.0435 | Val Loss: 0.0429


    Epoch 03: 100%|██████████| 794/794 [00:12<00:00, 63.70it/s]


    Epoch 03 | Train Loss: 0.0368 | Val Loss: 0.0381


    Epoch 04: 100%|██████████| 794/794 [00:12<00:00, 64.67it/s]


    Epoch 04 | Train Loss: 0.0333 | Val Loss: 0.0349


    Epoch 05: 100%|██████████| 794/794 [00:12<00:00, 65.96it/s]


    Epoch 05 | Train Loss: 0.0311 | Val Loss: 0.0318


    Epoch 06: 100%|██████████| 794/794 [00:12<00:00, 65.91it/s]


    Epoch 06 | Train Loss: 0.0296 | Val Loss: 0.0315


    Epoch 07: 100%|██████████| 794/794 [00:12<00:00, 63.92it/s]


    Epoch 07 | Train Loss: 0.0282 | Val Loss: 0.0306


    Epoch 08: 100%|██████████| 794/794 [00:12<00:00, 63.87it/s]


    Epoch 08 | Train Loss: 0.0273 | Val Loss: 0.0279


    Epoch 09: 100%|██████████| 794/794 [00:11<00:00, 66.54it/s]


    Epoch 09 | Train Loss: 0.0264 | Val Loss: 0.0275


    Epoch 10: 100%|██████████| 794/794 [00:12<00:00, 65.40it/s]


    Epoch 10 | Train Loss: 0.0258 | Val Loss: 0.0275


    Epoch 11: 100%|██████████| 794/794 [00:12<00:00, 65.44it/s]


    Epoch 11 | Train Loss: 0.0250 | Val Loss: 0.0269


    Epoch 12: 100%|██████████| 794/794 [00:11<00:00, 66.90it/s]


    Epoch 12 | Train Loss: 0.0246 | Val Loss: 0.0263


    Epoch 13: 100%|██████████| 794/794 [00:12<00:00, 65.99it/s]


    Epoch 13 | Train Loss: 0.0239 | Val Loss: 0.0263


    Epoch 14: 100%|██████████| 794/794 [00:11<00:00, 66.38it/s]


    Epoch 14 | Train Loss: 0.0233 | Val Loss: 0.0255


    Epoch 15: 100%|██████████| 794/794 [00:11<00:00, 66.35it/s]


    Epoch 15 | Train Loss: 0.0230 | Val Loss: 0.0259


    Epoch 16: 100%|██████████| 794/794 [00:12<00:00, 65.32it/s]


    Epoch 16 | Train Loss: 0.0227 | Val Loss: 0.0251


    Epoch 17: 100%|██████████| 794/794 [00:12<00:00, 65.14it/s]


    Epoch 17 | Train Loss: 0.0225 | Val Loss: 0.0248


    Epoch 18: 100%|██████████| 794/794 [00:12<00:00, 65.07it/s]


    Epoch 18 | Train Loss: 0.0219 | Val Loss: 0.0252


    Epoch 19: 100%|██████████| 794/794 [00:12<00:00, 65.92it/s]


    Epoch 19 | Train Loss: 0.0218 | Val Loss: 0.0265


    Epoch 20: 100%|██████████| 794/794 [00:11<00:00, 66.22it/s]


    Epoch 20 | Train Loss: 0.0215 | Val Loss: 0.0252
    Early stopping triggered at epoch 20
    [TCN] Test MAE: 845.14
    [TCN] Test RMSE: 1152.52




    CPU times: user 3min 20s, sys: 28.6 s, total: 3min 48s
    Wall time: 4min 23s


**TCN improves forecasts compared to LSTM.**

### Transformer
Transformers are attention-based models originally developed for natural language processing, however, they have proven effective for time series tasks. A transformer uses the self-attention mechanisms to weigh the importance of each timestep in the input sequence, which enables them to model both short- and long-range dependencies efficiently. Also, a transformer processes the entire sequence in parallel and is not limited by step-by-step recurrence or fixed convolutional windows.


```python
%%time

import torch
from models.tf import TF_ECF
from utils.training_utils import set_seed, train, evaluate
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils.visualization_utils import plot_actual_vs_pred

set_seed(42)

tf_model = TF_ECF(input_size=X_train.shape[2])

tf_model = train(
    tf_model,
    train_loader=train_dl,
    val_loader=val_dl,
    optimizer=torch.optim.Adam(tf_model.parameters(), lr=1e-3),
    loss_fn=torch.nn.MSELoss(),
    num_epochs=30,
    patience=3,
    device=device,
)

tf_y_true, tf_y_pred = evaluate(tf_model, test_dl, device)
tf_y_true = target_scaler.inverse_transform(tf_y_true.reshape(-1, 1)).flatten()
tf_y_pred = target_scaler.inverse_transform(tf_y_pred.reshape(-1, 1)).flatten()

tf_mae = mean_absolute_error(tf_y_true, tf_y_pred)
tf_rmse = root_mean_squared_error(tf_y_true, tf_y_pred)

print(f"[TF] Test MAE: {tf_mae:.2f}")
print(f"[TF] Test RMSE: {tf_rmse:.2f}")

plot_actual_vs_pred(tf_y_true, tf_y_pred, title="Transformer")

```

    Epoch 01: 100%|██████████| 794/794 [00:13<00:00, 59.50it/s]


    Epoch 01 | Train Loss: 0.0728 | Val Loss: 0.0378


    Epoch 02: 100%|██████████| 794/794 [00:13<00:00, 58.70it/s]


    Epoch 02 | Train Loss: 0.0367 | Val Loss: 0.0312


    Epoch 03: 100%|██████████| 794/794 [00:13<00:00, 60.47it/s]


    Epoch 03 | Train Loss: 0.0309 | Val Loss: 0.0282


    Epoch 04: 100%|██████████| 794/794 [00:13<00:00, 60.24it/s]


    Epoch 04 | Train Loss: 0.0278 | Val Loss: 0.0259


    Epoch 05: 100%|██████████| 794/794 [00:13<00:00, 60.66it/s]


    Epoch 05 | Train Loss: 0.0257 | Val Loss: 0.0242


    Epoch 06: 100%|██████████| 794/794 [00:13<00:00, 60.82it/s]


    Epoch 06 | Train Loss: 0.0239 | Val Loss: 0.0238


    Epoch 07: 100%|██████████| 794/794 [00:13<00:00, 59.54it/s]


    Epoch 07 | Train Loss: 0.0222 | Val Loss: 0.0216


    Epoch 08: 100%|██████████| 794/794 [00:13<00:00, 57.81it/s]


    Epoch 08 | Train Loss: 0.0207 | Val Loss: 0.0201


    Epoch 09: 100%|██████████| 794/794 [00:13<00:00, 58.57it/s]


    Epoch 09 | Train Loss: 0.0193 | Val Loss: 0.0200


    Epoch 10: 100%|██████████| 794/794 [00:13<00:00, 59.77it/s]


    Epoch 10 | Train Loss: 0.0181 | Val Loss: 0.0205


    Epoch 11: 100%|██████████| 794/794 [00:13<00:00, 59.39it/s]


    Epoch 11 | Train Loss: 0.0171 | Val Loss: 0.0175


    Epoch 12: 100%|██████████| 794/794 [00:13<00:00, 59.57it/s]


    Epoch 12 | Train Loss: 0.0160 | Val Loss: 0.0164


    Epoch 13: 100%|██████████| 794/794 [00:13<00:00, 59.73it/s]


    Epoch 13 | Train Loss: 0.0152 | Val Loss: 0.0168


    Epoch 14: 100%|██████████| 794/794 [00:13<00:00, 59.42it/s]


    Epoch 14 | Train Loss: 0.0147 | Val Loss: 0.0153


    Epoch 15: 100%|██████████| 794/794 [00:13<00:00, 59.15it/s]


    Epoch 15 | Train Loss: 0.0142 | Val Loss: 0.0153


    Epoch 16: 100%|██████████| 794/794 [00:13<00:00, 59.28it/s]


    Epoch 16 | Train Loss: 0.0139 | Val Loss: 0.0153


    Epoch 17: 100%|██████████| 794/794 [00:13<00:00, 59.55it/s]


    Epoch 17 | Train Loss: 0.0137 | Val Loss: 0.0142


    Epoch 18: 100%|██████████| 794/794 [00:13<00:00, 58.23it/s]


    Epoch 18 | Train Loss: 0.0133 | Val Loss: 0.0157


    Epoch 19: 100%|██████████| 794/794 [00:14<00:00, 55.93it/s]


    Epoch 19 | Train Loss: 0.0131 | Val Loss: 0.0161


    Epoch 20: 100%|██████████| 794/794 [00:13<00:00, 59.11it/s]


    Epoch 20 | Train Loss: 0.0126 | Val Loss: 0.0142
    Early stopping triggered at epoch 20
    [TF] Test MAE: 603.12
    [TF] Test RMSE: 837.75




    CPU times: user 4min 24s, sys: 33.2 s, total: 4min 57s
    Wall time: 4min 46s


**Transformer achieves most accurate forecasts.**

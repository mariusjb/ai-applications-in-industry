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
    CPU times: user 9.76 ms, sys: 14.6 ms, total: 24.4 ms
    Wall time: 1.08 s



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
    CPU times: user 93.1 ms, sys: 43.7 ms, total: 137 ms
    Wall time: 213 ms



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

# Define the window size and forecast steps, e.g., use the last 30 days to forecast the next 3 days
window_size = 24 * 30  # 24h * 30days
forecast_steps = 24 * 3  # 24h * 3days

X_train, y_train = create_sequences(train_scaled, window_size=window_size, forecast_steps=forecast_steps)
X_val, y_val = create_sequences(val_scaled, window_size=window_size, forecast_steps=forecast_steps)
X_test, y_test = create_sequences(test_scaled, window_size=window_size, forecast_steps=forecast_steps)

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

     11%|█         | 11090/100941 [00:07<00:46, 1916.92it/s]


```python
device = "mps"  # Defined once; change to "cpu" or - if available - "cuda"
```

### Long Short-Term Memory


```python
%%time

import torch
from models.lstm import LSTM_ECF
from utils.training_utils import set_seed, train, evaluate_multi_step
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils.visualization_utils import plot_multi_sample_forecasts

set_seed(42)

lstm_model = LSTM_ECF(input_size=X_train.shape[2], forecast_steps=forecast_steps)

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

lstm_y_true, lstm_y_pred = evaluate_multi_step(lstm_model, test_dl, device)
lstm_y_true = target_scaler.inverse_transform(lstm_y_true.reshape(-1, 1)).flatten()
lstm_y_pred = target_scaler.inverse_transform(lstm_y_pred.reshape(-1, 1)).flatten()

lstm_mae = mean_absolute_error(lstm_y_true, lstm_y_pred)
lstm_rmse = root_mean_squared_error(lstm_y_true, lstm_y_pred)

print(f"[LSTM] Test MAE: {lstm_mae:.2f}")
print(f"[LSTM] Test RMSE: {lstm_rmse:.2f}")

plot_multi_sample_forecasts(
    y_true=lstm_y_true.reshape(-1, forecast_steps),
    y_pred=lstm_y_pred.reshape(-1, forecast_steps),
    num_samples=2,
    title_prefix="LSTM",
)
```

    Epoch 01: 100%|██████████| 788/788 [02:07<00:00,  6.19it/s]


    Epoch 01 | Train Loss: 0.2324 | Val Loss: 0.2187


    Epoch 02: 100%|██████████| 788/788 [01:56<00:00,  6.74it/s]


    Epoch 02 | Train Loss: 0.1582 | Val Loss: 0.2320


    Epoch 03: 100%|██████████| 788/788 [01:57<00:00,  6.72it/s]


    Epoch 03 | Train Loss: 0.1360 | Val Loss: 0.2777


    Epoch 04: 100%|██████████| 788/788 [01:57<00:00,  6.70it/s]


    Epoch 04 | Train Loss: 0.1161 | Val Loss: 0.2749
    Early stopping triggered at epoch 4
    [LSTM] Test MAE: 2906.94
    [LSTM] Test RMSE: 4142.23






    CPU times: user 2min 28s, sys: 28.4 s, total: 2min 56s
    Wall time: 8min 49s


### Temporal Convolutional Network


```python
%%time

import torch
from models.tcn import TCN_ECF
from utils.training_utils import set_seed, train, evaluate_multi_step
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils.visualization_utils import plot_multi_sample_forecasts

set_seed(42)

tcn_model = TCN_ECF(
    input_size=X_train.shape[2], num_channels=[64] * 4, kernel_size=5, dropout=0.3, forecast_steps=forecast_steps
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

tcn_y_true, tcn_y_pred = evaluate_multi_step(tcn_model, test_dl, device)
tcn_y_true = target_scaler.inverse_transform(tcn_y_true.reshape(-1, 1)).flatten()
tcn_y_pred = target_scaler.inverse_transform(tcn_y_pred.reshape(-1, 1)).flatten()

tcn_mae = mean_absolute_error(tcn_y_true, tcn_y_pred)
tcn_rmse = root_mean_squared_error(tcn_y_true, tcn_y_pred)

print(f"[TCN] Test MAE: {tcn_mae:.2f}")
print(f"[TCN] Test RMSE: {tcn_rmse:.2f}")

plot_multi_sample_forecasts(
    y_true=tcn_y_true.reshape(-1, forecast_steps),
    y_pred=tcn_y_pred.reshape(-1, forecast_steps),
    num_samples=2,
    title_prefix="TCN"
)

```

    Epoch 01:   0%|          | 0/788 [00:00<?, ?it/s]

    Epoch 01: 100%|██████████| 788/788 [01:02<00:00, 12.69it/s]


    Epoch 01 | Train Loss: 0.2739 | Val Loss: 0.2124


    Epoch 02: 100%|██████████| 788/788 [01:00<00:00, 13.02it/s]


    Epoch 02 | Train Loss: 0.1843 | Val Loss: 0.2300


    Epoch 03: 100%|██████████| 788/788 [01:00<00:00, 13.03it/s]


    Epoch 03 | Train Loss: 0.1659 | Val Loss: 0.2476


    Epoch 04: 100%|██████████| 788/788 [01:00<00:00, 12.99it/s]


    Epoch 04 | Train Loss: 0.1525 | Val Loss: 0.2802
    Early stopping triggered at epoch 4
    [TCN] Test MAE: 2922.14
    [TCN] Test RMSE: 4102.02






    CPU times: user 1min 2s, sys: 21.2 s, total: 1min 23s
    Wall time: 4min 25s


### Transformer


```python
%%time

import torch
from models.tf import TF_ECF
from utils.training_utils import set_seed, train, evaluate_multi_step
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils.visualization_utils import plot_multi_sample_forecasts

set_seed(42)

tf_model = TF_ECF(input_size=X_train.shape[2], forecast_steps=forecast_steps)

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

tf_y_true, tf_y_pred = evaluate_multi_step(tf_model, test_dl, device)
tf_y_true = target_scaler.inverse_transform(tf_y_true.reshape(-1, 1)).flatten()
tf_y_pred = target_scaler.inverse_transform(tf_y_pred.reshape(-1, 1)).flatten()

tf_mae = mean_absolute_error(tf_y_true, tf_y_pred)
tf_rmse = root_mean_squared_error(tf_y_true, tf_y_pred)

print(f"[TF] Test MAE: {tf_mae:.2f}")
print(f"[TF] Test RMSE: {tf_rmse:.2f}")

plot_multi_sample_forecasts(
    y_true=tf_y_true.reshape(-1, forecast_steps),
    y_pred=tf_y_pred.reshape(-1, forecast_steps),
    num_samples=2,
    title_prefix="Transformer"
)

```

    Epoch 01: 100%|██████████| 788/788 [07:52<00:00,  1.67it/s]


    Epoch 01 | Train Loss: 0.2782 | Val Loss: 0.2078


    Epoch 02: 100%|██████████| 788/788 [07:54<00:00,  1.66it/s]


    Epoch 02 | Train Loss: 0.1630 | Val Loss: 0.2096


    Epoch 03: 100%|██████████| 788/788 [07:54<00:00,  1.66it/s]


    Epoch 03 | Train Loss: 0.1383 | Val Loss: 0.2299


    Epoch 04:  84%|████████▎ | 658/788 [06:50<01:21,  1.60it/s]



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    File <timed exec>:10


    File ~/Development/ai-applications-in-industry/energy_consumption/utils/training_utils.py:70, in train(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, device, patience, verbose, save_best)
         68     loss.backward()
         69     optimizer.step()
    ---> 70     train_loss += loss.item()
         72 # Validation
         73 model.eval()


    KeyboardInterrupt: 


**Note: Predictions reflect daily and hourly temporal patterns, as seen in the EDA.**

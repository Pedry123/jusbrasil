# Verificador de Citações Jurídicas

Projeto do desafio BRACIS 2026 × Jusbrasil. A aplicação recebe um parecer jurídico em `.txt`, identifica citações, classifica cada uma como `real`, `inventada` ou `incompleta` e permite baixar os resultados em JSON e CSV.

## Estrutura

O código executável está em `desafio-jusbrasil-bracis-2026/`. Os comandos abaixo devem ser executados a partir dessa pasta. O pipeline usa os arquivos Parquet em `data/`, os módulos de `src/` e o classificador com o índice FAISS em `models/`.

## Rodar localmente

1. Instale Python 3.12.
2. Abra um terminal na pasta do projeto:

   ```sh
   cd desafio-jusbrasil-bracis-2026
   ```

3. Crie e ative um ambiente virtual:

   ```sh
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

   No Windows PowerShell, ative com `.venv\Scripts\Activate.ps1`.

4. Instale as dependências:

   ```sh
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Gere os dados processados e treine o classificador. O treinamento baixa o modelo de embeddings na primeira execução:

   ```sh
   python -m src.merge_data
   python -m src.train
   ```

   Esses comandos criam/atualizam `data/merged_citacoes.parquet`, `data/canonical_base.parquet`, o classificador em `models/classifier.pt` e o índice `models/faiss.index`. Para reconstruir os dados, os arquivos de origem (`goldenset_offsets.csv`, `dataset.xlsx`, `txt/` e `desafio1_bracis.db`) precisam estar presentes.

6. Inicie a interface:

   ```sh
   streamlit run app/streamlit_app.py
   ```

7. Abra o endereço local informado pelo Streamlit, envie um `.txt` ou cole o parecer, clique em **Classificar citações** e baixe o JSON ou `submission.csv`.

Se os artefatos de treinamento já estiverem presentes, você pode pular o passo 5 e iniciar diretamente a interface.

## Rodar com Docker

O contexto de build é a pasta que contém o Dockerfile. A imagem instala as dependências, baixa o modelo de embeddings e treina o classificador durante o build.

```sh
cd desafio-jusbrasil-bracis-2026
docker build -t verificador-citacoes .
docker run --rm -p 8501:8501 verificador-citacoes
```

Depois, acesse [http://localhost:8501](http://localhost:8501). O build precisa de acesso à internet para baixar dependências Python e o modelo de embeddings. O `.dockerignore` reduz o contexto enviado ao Docker, mantendo apenas o código e os dois Parquet necessários ao treinamento da imagem.

## Reconstruir, avaliar e testar

A partir de `desafio-jusbrasil-bracis-2026/`, os dados e o modelo podem ser reconstruídos pelos comandos do passo 5. Para avaliar o pipeline e rodar os testes de regressão:

```sh
python -m src.evaluate --epochs 40
python -m unittest discover -s tests -v
```

`evaluate` registra os resultados em `data/evaluation.json` sem substituir o classificador salvo.

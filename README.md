# Verificador de Citações Jurídicas

Aplicação do desafio BRACIS 2026 × Jusbrasil. Recebe um parecer jurídico em `.txt`, identifica citações, classifica cada uma como `real`, `inventada` ou `incompleta` e permite baixar os resultados em JSON e CSV.

## Executar com Docker

### Requisitos

- Docker instalado e em execução.
- Acesso à internet durante a construção da imagem para baixar as dependências Python e o modelo de embeddings.

### 1. Abra um terminal na pasta do projeto

```sh
cd desafio-jusbrasil-bracis-2026
```

### 2. Construa a imagem

```sh
docker build -t verificador-citacoes .
```

O build instala as dependências, baixa o modelo de embeddings e treina o classificador. Essa etapa pode levar alguns minutos. É necessário refazê-la quando o código ou os dados de entrada da imagem forem alterados.

### 3. Inicie o container

```sh
docker run --rm -p 8501:8501 verificador-citacoes
```

O terminal exibirá os logs enquanto o container estiver em execução. Para encerrar, pressione `Ctrl+C`.

### 4. Use a aplicação

Abra [http://localhost:8501](http://localhost:8501), envie um parecer `.txt` ou cole o texto, clique em **Classificar citações** e baixe o JSON ou o `submission.csv`.

Se a porta `8501` já estiver ocupada, publique a porta do container em outra porta local:

```sh
docker run --rm -p 8502:8501 verificador-citacoes
```

Nesse caso, acesse [http://localhost:8502](http://localhost:8502).

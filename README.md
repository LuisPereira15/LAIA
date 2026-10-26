# LAIA — ambiente de desenvolvimento

Ambiente Python 3.12 em Docker para desenvolver e executar o código do projeto.
Requer Docker com suporte a contentores Linux e Docker Compose.

Na raiz do projeto, arrancar com:

```sh
docker compose up
```

O contentor `dev` fica ativo, à espera de comandos. Não arranca uma API,
não treina modelos e não publica nenhuma página ou porta. É normal não
aparecerem logs de aplicação. Para deixar o terminal livre, usar
`docker compose up -d`.

Noutro terminal, executar Python dentro do contentor:

```sh
docker compose exec dev python --version
docker compose exec dev python
```

Para executar um script que venha a ser criado, usar
`docker compose exec dev python caminho/do/script.py`.

A pasta do projeto está disponível em `/app`. O código e os dados mantêm-se
no computador; as alterações feitas em `/app` também se refletem nessa pasta.
Os datasets não são copiados para a imagem nem incluídos no Git.

Adicionar as dependências necessárias a `requirements.txt`, com versões
acordadas pela equipa, e reconstruir o ambiente:

```sh
docker compose up -d --build
```

Nesta versão inicial, o ficheiro de dependências está vazio: o ambiente
contém apenas Python e pip.

Para parar e remover o contentor, preservando os ficheiros do projeto:

```sh
docker compose down
```

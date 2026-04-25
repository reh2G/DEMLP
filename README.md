# DEMLP

## Ambiente e execução

Este projeto utiliza o `uv` para facilitar o gerenciamento de dependências e a execução do código.

Caso você ainda não tenha o `uv` instalado, consulte a documentação oficial para instalação:
https://github.com/astral-sh/uv

### 1. Instalar as dependências

```bash
uv sync
```

### 2. Configurar o ambiente

Necessário para uso de GPU durante o treinamento do modelo.

Este projeto utiliza TensorFlow com suporte a GPU via CUDA. As bibliotecas CUDA são instaladas como pacotes Python dentro do ambiente virtual, mas o sistema operacional precisa ser informado de onde encontrá-las.

O script abaixo configura automaticamente esse ambiente para a sua máquina:

```bash
./setup.sh
```

Caso ocorra um erro de permissão ao executar o script, rode:

```bash
chmod +x setup.sh
```

e tente novamente.

> [!IMPORTANT]
> Após rodar o `setup.sh`, abra um novo terminal no VS Code para que as configurações entrem em vigor.

### 3. Executar o projeto

```bash
uv run main.py
```

> [!NOTE]
> * O `setup.sh` precisa ser executado apenas **uma vez por máquina**, ou novamente caso o ambiente virtual seja recriado.
> * O arquivo `.vscode/settings.json` é gerado localmente pelo `setup.sh` e não é versionado no repositório. Cada colaborador terá sua própria configuração com os caminhos corretos para sua máquina.
> * O projeto foi desenvolvido e testado em **Linux com GPU NVIDIA**. Em outros sistemas operacionais, a configuração da GPU pode variar.

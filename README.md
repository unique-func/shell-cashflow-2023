<img src="./src/images/shell.png" align="center" height="150" width="160" />

# Shell Cashflow Datathon 2023
### Takım: Unique

Bu repository, `Unique` takımının `Shell Cashflow Datathon 2023` yarışması için hazırladığı konsept uygulamanın kaynak kodlarını içermektedir.

## Kullanım

### Gereksinimler
Kullandığınız `conda` paket yöneticisinin en son sürümü kurulu olmalıdır. Lütfen [bu linkteki](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) yönergeleri izleyin.

### Ortam Kurulumu
Projeyi herhangi bir problem yaşamadan çalıştırabilmek için lütfen aşağıdaki komutları kullanarak sanal geliştirme ortamınızı oluşturun:
```bash
conda env create --file env.yaml
conda activate shell
```

### Uygulamanın Çalıştırılması
Aşağıdaki kod ile tanımlayacağınız veri ile bütün süreci koşup çıktı alabilirsiniz.
```bash
streamlit run app.py
```

Ayrıca halihazırda deploy edilmiş örnek uygulamaya [buradan](https://huggingface.co/spaces/unique-func/shell-cashflow-datathon-2023) ulaşılabilirsiniz.
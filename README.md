# DiagnosIA Rim

Protótipo acadêmico de inteligência artificial para **triagem de doença renal crônica (DRC)**, desenvolvido para o seminário *Uso da Inteligência Artificial na Análise de Dados Clínicos para Diagnóstico Precoce*.

> **Aviso:** este protótipo tem fins exclusivamente educacionais. Não é um dispositivo médico regularizado (RDC Anvisa nº 657/2022) e não substitui avaliação médica.

## O que ele faz

A página estima a probabilidade de doença renal crônica a partir de dados clínicos e mostra quanto cada dado pesou na decisão. Há dois modelos:

| Modelo | Dados usados | Acurácia no teste | Sensibilidade | Especificidade |
| --- | --- | --- | --- | --- |
| Completo | 24 (inclui creatinina, ureia e exame de urina) | 99,2% | 98,7% | 100% |
| Triagem | 13 (consulta, pressão, glicemia e hemograma) | 95,0% | 96,0% | 93,3% |

O modelo de triagem não usa nenhum exame renal: a ideia é apontar, com exames baratos de rotina, quem deveria fazer os exames de função renal.

A página também permite ajustar o limiar de decisão e ver, em tempo real, como mudam sensibilidade, especificidade e a matriz de confusão.

## Como foi feito

1. **Dados:** base pública [Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) do UCI Machine Learning Repository, com 400 pacientes (250 com DRC e 150 sem), coletados em 2015 no Apollo Hospitals, Índia.
2. **Limpeza:** respostas em texto convertidas em números (não = 0, sim = 1; normal = 0, alterado = 1).
3. **Divisão:** 70% para treino (280 pacientes) e 30% para teste (120), estratificada.
4. **Dados ausentes:** só 158 pacientes tinham todos os dados. Os valores ausentes foram preenchidos com a mediana (numéricos) ou a moda (sim/não), calculadas apenas no treino.
5. **Modelo:** regressão logística com padronização das variáveis (scikit-learn).
6. **Validação:** validação cruzada de 5 partes no treino e avaliação final nos 120 pacientes de teste.
7. **Página:** os pesos do modelo foram embutidos em `index.html`, que refaz o cálculo no navegador. Nenhum dado é enviado para fora.

## Arquivos

- `index.html`: o protótipo interativo (funciona sozinho, sem servidor)
- `treino_renal.py`: código que treina os dois modelos e gera os pesos
- `relatorio.html`: relatório com o resultado dos dois modelos para todos os 400 pacientes da base indiana e os 200 de Bangladesh (validação externa), com filtros por grupo, diagnóstico e erros

Para reproduzir o treino, baixe a base no link acima, extraia o arquivo `chronic_kidney_disease_full.arff` e rode:

```bash
pip install numpy pandas scikit-learn
python treino_renal.py chronic_kidney_disease_full.arff
```

## Limitações

- O modelo completo usa creatinina e albumina, que fazem parte da própria definição da DRC, o que facilita o acerto.
- O grupo sem doença parece ser de pessoas saudáveis, o que torna a separação mais fácil do que na prática.
- Base pequena, de um único hospital indiano, sem validação em pacientes brasileiros.

## Fonte dos dados

RUBINI, L.; SOUNDARAPANDIAN, P.; ESWARAN, P. **Chronic Kidney Disease**. UCI Machine Learning Repository, 2015. DOI: https://doi.org/10.24432/C5G020. Licença CC BY 4.0.

Desenvolvido com apoio do assistente de IA Claude.

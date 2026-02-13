"""
Funções utilitárias para o projeto de Análise de Hemograma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def configurar_visualizacao():
    """
    Configura parâmetros padrão para visualizações
    """
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def valores_referencia_hemograma():
    """
    Retorna os valores de referência clínicos para hemograma
    
    Returns:
        dict: Dicionário com valores mínimos e máximos de cada marcador
    """
    referencia = {
        'hemoglobina': {
            'min': 12.0,  # g/dL
            'max': 16.0,
            'unidade': 'g/dL',
            'nome_completo': 'Hemoglobina'
        },
        'hematocrito': {
            'min': 36.0,  # %
            'max': 48.0,
            'unidade': '%',
            'nome_completo': 'Hematócrito'
        },
        'vcm': {
            'min': 80.0,  # fL
            'max': 100.0,
            'unidade': 'fL',
            'nome_completo': 'VCM (Volume Corpuscular Médio)'
        },
        'hcm': {
            'min': 27.0,  # pg
            'max': 32.0,
            'unidade': 'pg',
            'nome_completo': 'HCM (Hemoglobina Corpuscular Média)'
        },
        'chcm': {
            'min': 32.0,  # g/dL
            'max': 36.0,
            'unidade': 'g/dL',
            'nome_completo': 'CHCM (Concentração de Hemoglobina)'
        },
        'leucocitos': {
            'min': 4000,  # células/μL
            'max': 11000,
            'unidade': 'células/μL',
            'nome_completo': 'Leucócitos'
        },
        'plaquetas': {
            'min': 150000,  # células/μL
            'max': 400000,
            'unidade': 'células/μL',
            'nome_completo': 'Plaquetas'
        }
    }
    return referencia


def verificar_valores_alterados(df, coluna, valores_ref):
    """
    Verifica quantos valores estão fora da faixa de referência
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        coluna (str): Nome da coluna a verificar
        valores_ref (dict): Dicionário com 'min' e 'max'
    
    Returns:
        pd.Series: Série booleana indicando valores alterados
    """
    return (df[coluna] < valores_ref['min']) | (df[coluna] > valores_ref['max'])


def criar_flags_alteracoes(df, mapeamento_colunas=None):
    """
    Cria colunas de flags indicando se cada marcador está alterado
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        mapeamento_colunas (dict): Mapeamento de nomes de colunas para nomes padronizados
    
    Returns:
        pd.DataFrame: DataFrame com colunas de flags adicionadas
    """
    df_copy = df.copy()
    ref = valores_referencia_hemograma()
    
    # Usar mapeamento se fornecido, senão usar nomes padrão
    if mapeamento_colunas is None:
        mapeamento_colunas = {k: k for k in ref.keys()}
    
    for col_padrao, col_real in mapeamento_colunas.items():
        if col_real in df_copy.columns and col_padrao in ref:
            flag_col = f'{col_padrao}_alterado'
            df_copy[flag_col] = verificar_valores_alterados(
                df_copy, col_real, ref[col_padrao]
            )
    
    return df_copy


def resumo_estatistico(df, colunas):
    """
    Gera resumo estatístico detalhado
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        colunas (list): Lista de colunas para análise
    
    Returns:
        pd.DataFrame: DataFrame com estatísticas descritivas
    """
    stats = df[colunas].describe().T
    stats['missing'] = df[colunas].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df)) * 100
    return stats


def plotar_distribuicao_com_referencia(df, coluna, valores_ref, ax=None):
    """
    Plota histograma com linhas de referência clínica
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        coluna (str): Nome da coluna
        valores_ref (dict): Valores de referência (min, max)
        ax: Eixo matplotlib (opcional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograma
    ax.hist(df[coluna].dropna(), bins=50, alpha=0.7, edgecolor='black')
    
    # Linhas de referência
    ax.axvline(valores_ref['min'], color='red', linestyle='--', 
               linewidth=2, label=f'Mín: {valores_ref["min"]}')
    ax.axvline(valores_ref['max'], color='red', linestyle='--', 
               linewidth=2, label=f'Máx: {valores_ref["max"]}')
    
    ax.set_xlabel(f'{valores_ref["nome_completo"]} ({valores_ref["unidade"]})')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição de {valores_ref["nome_completo"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def salvar_figura(fig, nome_arquivo, dpi=300):
    """
    Salva figura com alta resolução
    
    Args:
        fig: Figura matplotlib
        nome_arquivo (str): Nome do arquivo (sem extensão)
        dpi (int): Resolução da imagem
    """
    fig.savefig(f'reports/figures/{nome_arquivo}.png', 
                dpi=dpi, bbox_inches='tight')
    print(f"✓ Figura salva: reports/figures/{nome_arquivo}.png")


def imprimir_resumo_dataset(df):
    """
    Imprime resumo completo do dataset
    
    Args:
        df (pd.DataFrame): DataFrame a analisar
    """
    print("=" * 60)
    print("📊 RESUMO DO DATASET")
    print("=" * 60)
    print(f"\n🔢 Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"\n📋 Colunas: {', '.join(df.columns.tolist())}")
    print(f"\n💾 Uso de memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n❓ Valores faltantes:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"   • {col}: {count:,} ({pct:.2f}%)")
    else:
        print("   ✓ Nenhum valor faltante!")
    print("\n" + "=" * 60)

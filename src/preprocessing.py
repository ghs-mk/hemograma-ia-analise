"""
Funções de pré-processamento de dados para análise de hemograma
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer


class PreprocessadorHemograma:
    """
    Classe para pré-processamento de dados de hemograma
    """
    
    def __init__(self, estrategia_scaler='standard'):
        """
        Inicializa o preprocessador
        
        Args:
            estrategia_scaler (str): 'standard' ou 'robust'
        """
        self.scaler = StandardScaler() if estrategia_scaler == 'standard' else RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.colunas_numericas = None
        self.colunas_originais = None
        
    def tratar_valores_faltantes(self, df, estrategia='median'):
        """
        Trata valores faltantes no dataset
        
        Args:
            df (pd.DataFrame): DataFrame original
            estrategia (str): 'median', 'mean', 'drop'
        
        Returns:
            pd.DataFrame: DataFrame sem valores faltantes
        """
        df_copy = df.copy()
        
        if estrategia == 'drop':
            linhas_antes = len(df_copy)
            df_copy = df_copy.dropna()
            linhas_depois = len(df_copy)
            print(f"✓ Removidas {linhas_antes - linhas_depois} linhas com valores faltantes")
        
        elif estrategia in ['median', 'mean']:
            colunas_numericas = df_copy.select_dtypes(include=[np.number]).columns
            
            for col in colunas_numericas:
                missing_count = df_copy[col].isnull().sum()
                if missing_count > 0:
                    if estrategia == 'median':
                        valor_preenchimento = df_copy[col].median()
                    else:
                        valor_preenchimento = df_copy[col].mean()
                    
                    df_copy[col].fillna(valor_preenchimento, inplace=True)
                    print(f"✓ Preenchidos {missing_count} valores em '{col}' com {estrategia}: {valor_preenchimento:.2f}")
        
        return df_copy
    
    def detectar_outliers_iqr(self, df, colunas, multiplicador=1.5):
        """
        Detecta outliers usando método IQR (Interquartile Range)
        
        Args:
            df (pd.DataFrame): DataFrame
            colunas (list): Colunas para detectar outliers
            multiplicador (float): Multiplicador do IQR (padrão: 1.5)
        
        Returns:
            pd.DataFrame: DataFrame com coluna indicando outliers
        """
        df_copy = df.copy()
        df_copy['is_outlier'] = False
        
        for col in colunas:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - multiplicador * IQR
            limite_superior = Q3 + multiplicador * IQR
            
            outliers = (df_copy[col] < limite_inferior) | (df_copy[col] > limite_superior)
            df_copy['is_outlier'] = df_copy['is_outlier'] | outliers
            
            n_outliers = outliers.sum()
            if n_outliers > 0:
                print(f"✓ {n_outliers} outliers detectados em '{col}'")
        
        total_outliers = df_copy['is_outlier'].sum()
        print(f"\n📊 Total de linhas com outliers: {total_outliers} ({total_outliers/len(df_copy)*100:.2f}%)")
        
        return df_copy
    
    def tratar_outliers(self, df, colunas, metodo='winsorize', limite_percentil=0.05):
        """
        Trata outliers usando diferentes métodos
        
        Args:
            df (pd.DataFrame): DataFrame
            colunas (list): Colunas para tratar
            metodo (str): 'winsorize', 'cap', 'remove'
            limite_percentil (float): Percentil para winsorização (0.05 = 5% e 95%)
        
        Returns:
            pd.DataFrame: DataFrame com outliers tratados
        """
        df_copy = df.copy()
        
        for col in colunas:
            if metodo == 'winsorize':
                # Substitui valores extremos pelos percentis
                lower = df_copy[col].quantile(limite_percentil)
                upper = df_copy[col].quantile(1 - limite_percentil)
                
                df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
                print(f"✓ Winsorização aplicada em '{col}' (limites: {lower:.2f} - {upper:.2f})")
            
            elif metodo == 'cap':
                # Cap usando IQR
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
                print(f"✓ Capping aplicado em '{col}' (limites: {lower:.2f} - {upper:.2f})")
        
        return df_copy
    
    def padronizar_dados(self, df, colunas=None, fit=True):
        """
        Padroniza os dados (z-score normalization)
        
        Args:
            df (pd.DataFrame): DataFrame
            colunas (list): Colunas para padronizar (None = todas numéricas)
            fit (bool): Se True, ajusta o scaler. Se False, usa scaler já ajustado
        
        Returns:
            pd.DataFrame: DataFrame padronizado
        """
        df_copy = df.copy()
        
        if colunas is None:
            colunas = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        self.colunas_numericas = colunas
        
        if fit:
            df_copy[colunas] = self.scaler.fit_transform(df_copy[colunas])
            print(f"✓ Dados padronizados usando {type(self.scaler).__name__}")
        else:
            df_copy[colunas] = self.scaler.transform(df_copy[colunas])
            print(f"✓ Dados transformados usando scaler existente")
        
        return df_copy
    
    def criar_features_derivadas(self, df):
        """
        Cria features derivadas a partir dos dados originais
        
        Args:
            df (pd.DataFrame): DataFrame original
        
        Returns:
            pd.DataFrame: DataFrame com features adicionais
        """
        df_copy = df.copy()
        
        # Exemplo de features derivadas (adaptar conforme colunas do dataset)
        # Estas são sugestões - você pode criar outras baseadas no seu dataset
        
        # Proporções importantes
        if 'hemoglobina' in df_copy.columns and 'hematocrito' in df_copy.columns:
            df_copy['hb_ht_ratio'] = df_copy['hemoglobina'] / (df_copy['hematocrito'] + 0.001)
            print("✓ Feature criada: hb_ht_ratio (Hemoglobina/Hematócrito)")
        
        # Índices calculados
        if 'vcm' in df_copy.columns and 'hcm' in df_copy.columns:
            df_copy['vcm_hcm_ratio'] = df_copy['vcm'] / (df_copy['hcm'] + 0.001)
            print("✓ Feature criada: vcm_hcm_ratio")
        
        return df_copy
    
    def pipeline_completo(self, df, colunas_numericas=None, 
                         tratar_missing=True, tratar_outliers_flag=True,
                         padronizar=True, criar_features=True):
        """
        Executa pipeline completo de pré-processamento
        
        Args:
            df (pd.DataFrame): DataFrame original
            colunas_numericas (list): Colunas numéricas a processar
            tratar_missing (bool): Tratar valores faltantes
            tratar_outliers_flag (bool): Tratar outliers
            padronizar (bool): Padronizar dados
            criar_features (bool): Criar features derivadas
        
        Returns:
            pd.DataFrame: DataFrame processado
        """
        print("\n" + "="*60)
        print("🔧 INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        print("="*60 + "\n")
        
        df_processado = df.copy()
        
        # 1. Tratar valores faltantes
        if tratar_missing:
            print("📝 Etapa 1: Tratamento de valores faltantes")
            df_processado = self.tratar_valores_faltantes(df_processado, estrategia='median')
            print()
        
        # 2. Criar features derivadas (antes de tratar outliers)
        if criar_features:
            print("📝 Etapa 2: Criação de features derivadas")
            df_processado = self.criar_features_derivadas(df_processado)
            print()
        
        # 3. Tratar outliers
        if tratar_outliers_flag:
            print("📝 Etapa 3: Tratamento de outliers")
            if colunas_numericas is None:
                colunas_numericas = df_processado.select_dtypes(include=[np.number]).columns.tolist()
            df_processado = self.tratar_outliers(df_processado, colunas_numericas, metodo='winsorize')
            print()
        
        # 4. Padronização (último passo)
        if padronizar:
            print("📝 Etapa 4: Padronização dos dados")
            df_processado = self.padronizar_dados(df_processado, colunas_numericas)
            print()
        
        print("="*60)
        print("✅ PIPELINE DE PRÉ-PROCESSAMENTO CONCLUÍDO")
        print("="*60 + "\n")
        
        return df_processado


def dividir_treino_teste(df, test_size=0.2, random_state=42):
    """
    Divide dados em treino e teste
    
    Args:
        df (pd.DataFrame): DataFrame completo
        test_size (float): Proporção do teste (0.2 = 20%)
        random_state (int): Seed para reprodutibilidade
    
    Returns:
        tuple: (df_treino, df_teste)
    """
    from sklearn.model_selection import train_test_split
    
    df_treino, df_teste = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"✓ Dados divididos:")
    print(f"  • Treino: {len(df_treino):,} linhas ({(1-test_size)*100:.0f}%)")
    print(f"  • Teste: {len(df_teste):,} linhas ({test_size*100:.0f}%)")
    
    return df_treino, df_teste

"""
Modelos de Machine Learning para análise de hemograma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from scipy.stats import zscore


class DetectorAnomalias:
    """
    Classe para detecção de anomalias em dados de hemograma
    """
    
    def __init__(self, metodo='isolation_forest', contamination=0.1):
        """
        Inicializa detector de anomalias
        
        Args:
            metodo (str): 'isolation_forest' ou 'zscore'
            contamination (float): Proporção esperada de anomalias (0.1 = 10%)
        """
        self.metodo = metodo
        self.contamination = contamination
        self.modelo = None
        
        if metodo == 'isolation_forest':
            self.modelo = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
    
    def detectar(self, X):
        """
        Detecta anomalias nos dados
        
        Args:
            X (pd.DataFrame ou np.array): Features
        
        Returns:
            np.array: Array com -1 para anomalias e 1 para normais
        """
        if self.metodo == 'isolation_forest':
            predicoes = self.modelo.fit_predict(X)
            scores = self.modelo.score_samples(X)
            
            n_anomalias = (predicoes == -1).sum()
            print(f"✓ Anomalias detectadas: {n_anomalias} ({n_anomalias/len(X)*100:.2f}%)")
            
            return predicoes, scores
        
        elif self.metodo == 'zscore':
            # Calcula z-score para cada coluna
            z_scores = np.abs(zscore(X, nan_policy='omit'))
            
            # Considera anomalia se qualquer feature tiver |z| > 3
            anomalias = (z_scores > 3).any(axis=1)
            predicoes = np.where(anomalias, -1, 1)
            
            n_anomalias = (predicoes == -1).sum()
            print(f"✓ Anomalias detectadas (Z-score): {n_anomalias} ({n_anomalias/len(X)*100:.2f}%)")
            
            return predicoes, z_scores.max(axis=1)
    
    def plotar_anomalias(self, X, predicoes, scores, feature_x=0, feature_y=1, 
                        nomes_features=None):
        """
        Visualiza anomalias em 2D
        
        Args:
            X (pd.DataFrame ou np.array): Features
            predicoes (np.array): Predições (-1 ou 1)
            scores (np.array): Scores de anomalia
            feature_x (int): Índice da feature para eixo X
            feature_y (int): Índice da feature para eixo Y
            nomes_features (list): Nomes das features
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Converter para numpy se necessário
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if nomes_features is None:
                nomes_features = X.columns.tolist()
        else:
            X_array = X
        
        if nomes_features is None:
            nomes_features = [f'Feature {i}' for i in range(X_array.shape[1])]
        
        # Gráfico 1: Scatter plot colorido por anomalia
        normais = predicoes == 1
        anomalias = predicoes == -1
        
        ax1.scatter(X_array[normais, feature_x], X_array[normais, feature_y], 
                   c='blue', label='Normal', alpha=0.6, s=50)
        ax1.scatter(X_array[anomalias, feature_x], X_array[anomalias, feature_y], 
                   c='red', label='Anomalia', alpha=0.8, s=100, marker='x')
        ax1.set_xlabel(nomes_features[feature_x])
        ax1.set_ylabel(nomes_features[feature_y])
        ax1.set_title('Detecção de Anomalias')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Distribuição dos scores
        ax2.hist(scores[normais], bins=50, alpha=0.7, label='Normal', color='blue')
        ax2.hist(scores[anomalias], bins=50, alpha=0.7, label='Anomalia', color='red')
        ax2.set_xlabel('Anomaly Score')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição dos Scores de Anomalia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class ClusterizadorPerfis:
    """
    Classe para clusterização de perfis hematológicos
    """
    
    def __init__(self, n_clusters=3, metodo='kmeans'):
        """
        Inicializa clusterizador
        
        Args:
            n_clusters (int): Número de clusters
            metodo (str): 'kmeans' ou 'hierarchical'
        """
        self.n_clusters = n_clusters
        self.metodo = metodo
        self.modelo = None
        self.labels_ = None
        
        if metodo == 'kmeans':
            self.modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif metodo == 'hierarchical':
            self.modelo = AgglomerativeClustering(n_clusters=n_clusters)
    
    def encontrar_n_clusters_otimo(self, X, max_clusters=10):
        """
        Encontra número ótimo de clusters usando método do cotovelo e silhouette
        
        Args:
            X (pd.DataFrame ou np.array): Features
            max_clusters (int): Número máximo de clusters a testar
        
        Returns:
            dict: Dicionário com scores para cada n_clusters
        """
        inertias = []
        silhouette_scores = []
        db_scores = []
        
        range_clusters = range(2, max_clusters + 1)
        
        for k in range_clusters:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            db_scores.append(davies_bouldin_score(X, labels))
        
        # Plotar métricas
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Método do cotovelo
        axes[0].plot(range_clusters, inertias, 'bo-')
        axes[0].set_xlabel('Número de Clusters')
        axes[0].set_ylabel('Inércia')
        axes[0].set_title('Método do Cotovelo')
        axes[0].grid(True)
        
        # Silhouette Score (maior é melhor)
        axes[1].plot(range_clusters, silhouette_scores, 'go-')
        axes[1].set_xlabel('Número de Clusters')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score (↑ melhor)')
        axes[1].grid(True)
        
        # Davies-Bouldin Index (menor é melhor)
        axes[2].plot(range_clusters, db_scores, 'ro-')
        axes[2].set_xlabel('Número de Clusters')
        axes[2].set_ylabel('Davies-Bouldin Index')
        axes[2].set_title('Davies-Bouldin Index (↓ melhor)')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Recomendar melhor número de clusters
        melhor_silhouette = range_clusters[np.argmax(silhouette_scores)]
        melhor_db = range_clusters[np.argmin(db_scores)]
        
        print(f"\n📊 Recomendações:")
        print(f"  • Melhor Silhouette Score: {melhor_silhouette} clusters")
        print(f"  • Melhor Davies-Bouldin: {melhor_db} clusters")
        
        return {
            'n_clusters': list(range_clusters),
            'inertia': inertias,
            'silhouette': silhouette_scores,
            'davies_bouldin': db_scores,
            'recomendacao_silhouette': melhor_silhouette,
            'recomendacao_db': melhor_db,
            'figura': fig
        }
    
    def treinar(self, X):
        """
        Treina o modelo de clusterização
        
        Args:
            X (pd.DataFrame ou np.array): Features
        
        Returns:
            np.array: Labels dos clusters
        """
        self.labels_ = self.modelo.fit_predict(X)
        
        # Calcular métricas
        silhouette = silhouette_score(X, self.labels_)
        db_score = davies_bouldin_score(X, self.labels_)
        
        print(f"\n✓ Clusterização concluída!")
        print(f"  • Método: {self.metodo}")
        print(f"  • Número de clusters: {self.n_clusters}")
        print(f"  • Silhouette Score: {silhouette:.3f}")
        print(f"  • Davies-Bouldin Index: {db_score:.3f}")
        
        # Distribuição dos clusters
        unique, counts = np.unique(self.labels_, return_counts=True)
        print(f"\n📊 Distribuição dos clusters:")
        for cluster, count in zip(unique, counts):
            print(f"  • Cluster {cluster}: {count} amostras ({count/len(self.labels_)*100:.1f}%)")
        
        return self.labels_
    
    def visualizar_clusters_pca(self, X, labels=None):
        """
        Visualiza clusters usando PCA para redução de dimensionalidade
        
        Args:
            X (pd.DataFrame ou np.array): Features
            labels (np.array): Labels dos clusters (usa self.labels_ se None)
        """
        if labels is None:
            labels = self.labels_
        
        # Redução para 2D com PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=labels, cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='black')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variância)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variância)')
        ax.set_title('Visualização dos Clusters (PCA)')
        
        plt.colorbar(scatter, label='Cluster')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, pca
    
    def caracterizar_clusters(self, X, labels=None, nomes_features=None):
        """
        Caracteriza cada cluster baseado nas médias das features
        
        Args:
            X (pd.DataFrame ou np.array): Features
            labels (np.array): Labels dos clusters
            nomes_features (list): Nomes das features
        
        Returns:
            pd.DataFrame: DataFrame com características médias por cluster
        """
        if labels is None:
            labels = self.labels_
        
        if isinstance(X, pd.DataFrame):
            df_caracteristicas = X.copy()
            if nomes_features is None:
                nomes_features = X.columns.tolist()
        else:
            if nomes_features is None:
                nomes_features = [f'Feature_{i}' for i in range(X.shape[1])]
            df_caracteristicas = pd.DataFrame(X, columns=nomes_features)
        
        df_caracteristicas['Cluster'] = labels
        
        # Calcular médias por cluster
        caracteristicas = df_caracteristicas.groupby('Cluster').mean()
        
        print("\n📊 Características médias por cluster:")
        print(caracteristicas.round(2))
        
        return caracteristicas


class ClassificadorHemograma:
    """
    Classe para classificação de hemogramas (Normal vs Alterado)
    """
    
    def __init__(self, modelo='random_forest'):
        """
        Inicializa classificador
        
        Args:
            modelo (str): Tipo de modelo ('random_forest', 'logistic')
        """
        self.modelo_tipo = modelo
        
        if modelo == 'random_forest':
            self.modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def treinar(self, X_train, y_train):
        """
        Treina o modelo
        """
        self.modelo.fit(X_train, y_train)
        print(f"✓ Modelo treinado: {self.modelo_tipo}")
    
    def avaliar(self, X_test, y_test):
        """
        Avalia o modelo
        """
        y_pred = self.modelo.predict(X_test)
        
        print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
        print("="*60)
        print(classification_report(y_test, y_pred))
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusão')
        
        return y_pred, cm, fig
    
    def feature_importance(self, nomes_features):
        """
        Mostra importância das features
        """
        if hasattr(self.modelo, 'feature_importances_'):
            importancias = self.modelo.feature_importances_
            
            df_imp = pd.DataFrame({
                'Feature': nomes_features,
                'Importância': importancias
            }).sort_values('Importância', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(df_imp['Feature'], df_imp['Importância'])
            ax.set_xlabel('Importância')
            ax.set_title('Feature Importance')
            ax.invert_yaxis()
            
            print("\n📊 TOP 10 Features Mais Importantes:")
            print(df_imp.head(10).to_string(index=False))
            
            return df_imp, fig

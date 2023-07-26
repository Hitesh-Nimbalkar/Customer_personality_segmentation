
import seaborn as sns
import matplotlib.pyplot as plt

class train_plot:
    def __init__(self,df) -> None:
        self.df=df.copy()
        
        
    def plot_cluster_boxplot(self,  y_col='Total_Amount',cluster_col='Customer_cluster'):
        df = self.df
        
        # Set the style of the plot
        sns.set(style="whitegrid")

        # Create the box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Swarm plot
        sns.swarmplot(x=df[cluster_col], y=df[y_col], color="gray", alpha=0.5, ax=ax)

        # Box plot
        sns.boxenplot(x=df[cluster_col], y=df[y_col], palette="Set2", ax=ax)
        
        # Title
        ax.set_title("Boxplot of Customers Clusters", pad=10, size=15)

        # Adjust the plot layout
        plt.tight_layout()

        # Show the plot
        plt.show()
        
        
        
        

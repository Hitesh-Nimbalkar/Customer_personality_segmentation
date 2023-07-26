
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
        
        
        
        
'''
@app.route('/train_plot', methods=['GET', 'POST'])
def train():
    try:
        
        # Load YAML data from file
        with open('cluster_label.yaml', 'r') as file:
            yaml_data = yaml.safe_load(file)

        # Convert YAML data to dictionary
        data_dict = {int(key): value for key, value in yaml_data.items()}
        
        # Displaying disct data 
        logging.info(f" Data in cluster report : {data_dict}")

        # Create a DataFrame with the "prediction" column
        prediction_df = pd.read_csv(csv_path)

        # Assign the values from data_dict to the "prediction" column of your existing DataFrame
        prediction_df['Customer_cluster'] = prediction_df['cluster'].map(data_dict)
        
        
        
        
        
        output = "Plot Prediction Done "
        return render_template("train_plot.html", prediction_result=output,prediction_type='batch',hist_plot=destination_static_folder,box_plot=)

        
        
    except Exception as e:
        logging.error(str(e))
        error_message = "An error occurred Please try again."
        return render_template('train_plot.html', error=error_message)

'''
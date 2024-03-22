from setuptools import setup, find_packages

setup(
    name='factor_graph_insights',
    version='0.1',
    description="Analysis factor graphs from droid slam for covariance calculation",
    author="Jagatpreet Singh Nir",
    author_email="nir.j@northeastern.edu",
    packages=find_packages(),
    install_requires=["gtsam", "matplotlib" , "pathlib" , "typing"],
    keywords=["python", "factor_graph", "covariance"],
    classifiers=[],
    entry_points={
        'console_scripts':[
          'visualize_covariance_plots=factor_graph_insights.main:visualize_factor_graphs',
          'animate_factor_graphs=factor_graph_insights.main:animate',
          'process_complete_factor_graph_data=factor_graph_insights.main:process_factor_graphs',  
        ],
    }
    
)

print(f'{find_packages()}')

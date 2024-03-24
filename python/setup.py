from setuptools import setup, find_packages

setup(
    name='factor_graph_insights',
    version='1.0',
    description="Analysis factor graphs from droid slam for covariance calculation",
    author="Jagatpreet Singh Nir",
    author_email="nir.j@northeastern.edu",
    packages=find_packages(),
    install_requires=["gtsam", "matplotlib" , "pathlib" , "typing"],
    keywords=["python", "factor_graph", "covariance"],
    classifiers=[],
    entry_points={
        'console_scripts':[
          'fg_file_processor=factor_graph_insights.main_cli:handle_file',
          'fg_covariance_animator=factor_graph_insights.main_cli:animate',
          'fg_batch_processor=factor_graph_insights.main_cli:handle_batch',
          'fg_image_viewer=factor_graph_insights.main_cli:images_in_factor_graph',
          'run_simulation=factor_graph_insights.simulation:main',
        ],
    }
    
)

print(f'{find_packages()}')

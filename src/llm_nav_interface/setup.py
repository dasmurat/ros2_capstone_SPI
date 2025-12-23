from setuptools import find_packages, setup

package_name = 'llm_nav_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
            ('share/' + package_name, ['llm_nav_interface/benchmarking/warehouse_locations.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='muratpc',
    maintainer_email='muratpc@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_nav_init = llm_nav_interface.llm_nav_init:main',
            'llm_nav_goal_sender = llm_nav_interface.llm_nav_goal_sender:main'
        ],
    },
)

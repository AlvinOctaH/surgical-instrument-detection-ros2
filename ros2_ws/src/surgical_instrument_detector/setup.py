from setuptools import find_packages, setup

package_name = 'surgical_instrument_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alvin',
    maintainer_email='alvin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'detector_node   = surgical_instrument_detector.detector_node:main',
        'test_publisher  = surgical_instrument_detector.test_publisher:main',
    	],
    },
)

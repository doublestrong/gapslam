from setuptools import setup, find_packages

setup(
    name='gapslam',
    version='0.0',
    # packages=find_packages(),
    packages=['slam', 'stats', 'utils', 'factors', 'sampler', 'geometry','adaptive_inference'],
    package_dir={'': 'gapslam'},
    url='',
    license='MIT',
    author='Qiangqiang Huang',
    author_email='qiangqiang.huang.me@gmail.com',
    description='Blending Gaussian approximation and particle filters for real-time non-Gaussian SLAM'
)

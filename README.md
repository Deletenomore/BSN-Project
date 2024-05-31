# BSN Project 2024
## Objective: This project aims to utilize AI tools such as ChatGPT to efficiently seek out a set of practical algorithms for specific medical sensor data analysis. 
## This repo is the prototype of data analysis models constructed by ChatGPT.

## Dataset
Zdemir,Ahmet and Barshan,Billur. (2018). Simulated Falls and Daily Living Activities Data Set. UCI Machine Learning Repository. https://doi.org/10.24432/C52028.

For demonstrating purpose, we selected 10 Fall actions for building up our models. They are properly labelled and the models.
- Number of Instance: 300
  * 6 Subjects(3 males and 3 females) x 5 tests x 10 fall actions. 
- Number of Attributes:
  * 10 Falls Actions
  * 6 Sensors each includes 3 axis Accelerometer, Gyroscope and Magnetometer.
- Each subject performed at least 5 tests and some of activities contain 6 tests of each action. 
- Each test contains 6 sensor's data
    * Head sensor
    * Chest sensor
    * Waist sensor
    * Right wrist sensor
    * Right thigh sensor
    * Right ankle sensor 
  - Labels:
    * 901-front-sitting
    * 902-front-protecting
    * 903-front-knees
    * 904-front-knees-lying
    * 905-front-quick-recovery
    * 906-front-slow-recovery
    * 907-front-right
    * 908-front-left
    * 909-back-sitting
    * 910-back-knees


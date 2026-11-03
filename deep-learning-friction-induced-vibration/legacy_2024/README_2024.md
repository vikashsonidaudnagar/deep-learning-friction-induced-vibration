Optimization of Friction-Induced Vibration System Using Stochastic Gradient Descent with Gradient Clipping


Project Details:
Developed a numerical solver to simulate a friction-induced vibration system using a second-order differential equation.
Implemented an exponential friction model to analyze energy loss due to friction in a mass-spring-damper system.
Optimized system parameters (damping coefficient, stiffness, friction coefficient, and friction sensitivity) to minimize energy loss.
Replaced Bayesian optimization with Stochastic Gradient Descent (SGD) with Gradient Clipping for improved stability and reduced likelihood of exploding gradients during the optimization process.
Utilized the R² score as a metric to evaluate the accuracy of the model's predictions against observed data.
Visualized key system dynamics, including displacement, velocity, acceleration, external force, relative velocity, and total energy, under optimized conditions.
Tools Used:
Python: For implementing the numerical solver, optimization, and simulation of the vibration system.
SciPy: Used solve_ivp for solving the ordinary differential equations (ODEs) that govern the system dynamics.
Scikit-Optimize (skopt): Employed for the optimization process, specifically for implementing Stochastic Gradient Descent with Gradient Clipping.
Matplotlib: For visualizing the simulation results, including displacement, velocity, acceleration, external force, relative velocity, and total energy.
Scikit-Learn (sklearn): Used r2_score to calculate the R² value, assessing the accuracy of the model’s predictions.


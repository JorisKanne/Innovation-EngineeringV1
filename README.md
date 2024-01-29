# Innovation-Engineering
## Neural Network for Predicting if Hamstring Graft is a Succesful Surgery.
The documentation of Team Blue.

In the Netherlands Hamstring (HS) Graft is the preferred graft to use when someone, raptures their ACL. When using this procedure, a person's demographical data is not considered. Everyone gets the same treatment, but there are more types of grafts. What if we can predict if the HS Graft is succesful or not, based on that surgeons can choose to use another graft that fits better for the person.

We used a Neural Network for our predictions. The Network is built from 5 inputs (female, male, age, height, weight), 2 hidden layers (Sigmoid, ReLU) and SGD as optimizer.
The predictions are based on a person's Lunge Time (the time it takes to put your knee to the ground).

A dashboard is included, this can be used by surgeons when a patient is in for a consultant appointment. On the dashboard, the demographical data of the person will be filled in, after analysing the predicted lunge time of the person 6 months after surgery with the right kind of physiotherapy is shown. Besides that, the HS Graft will be shown as successful or not, and some comparison to the population you as a person fall into is shown. The MRI taken of the knee and previous injury are also shown. 

# RecSys
This is a movie recommendation system based on Vue2.0, SpringBoot, MySQL, Mybatis, MongoDB, MinIO and Pytorch.

# Component
The system can be divided into two parts according to its functionality: the fontend (user end) and the backend (management end)：


(1) User end. The main users of the user end are website users and visitors. After registering, users can access the website, browse movie information and comment on movies. Movie information is divided into recommendations and film libraries. Users can search for movies based on keywords in film libraries. In addition, users can freely modify their personal information and passwords within the website and view their browsing history and system announcements.


(2) Backend management end. The main users of the backend management end are administrators. Administrators with different permissions can perform different operations. The allocation and recovery of permissions are carried out by the super administrator through the permission management function. The main operations of the management end include user information maintenance (deleting users, adding new users, modifying user information and searching for user information); comment management (batch adding comments, modifying or deleting their own published comments); movie information maintenance (deleting movies, adding new movies, editing and modifying movie information and searching for movies based on keywords); announcement management (batch publishing announcements to users, modifying or deleting announcements); dynamic permission management (granting and revoking permissions), viewing system data reports (such as user occupation, age and gender distribution; active users; popular movies); homepage carousel image management (such as adding carousel images, invalidating them and modifying them).

The recommendation model is implemented based on Pytorch. The relevant model selection is located in the RecSys folder and includes ItemCF, UserCF, MF, GMF(He et al., WWW’17), NeuCF(He et al., WWW’17), ConvNCF(He et al., IJCAI’18) and ENMF(Zhang et al., TOIS 2020).

# Note
If you want to run this project, the paths of each service need to be modified.

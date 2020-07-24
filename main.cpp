#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setGeometry(WINDOW_X,WINDOW_Y,WINDOW_WIDTH,WINDOW_HEIGHT);
    w.show();


    return a.exec();
}

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QCoreApplication>

#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QDebug>
#include <QToolBar>
#include <QPushButton>
#include <QStatusBar>
#include <QLabel>
#include <QTextEdit>
#include <QDockWidget>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QMenuBar *mBar = menuBar();
    QMenu *pFile = mBar->addMenu("檔案");

    m_button = new QPushButton("Button 1");
    m_button->setGeometry(QRect(QPoint(1000, 1000),QSize(2000, 5000)));
    connect(m_button, SIGNAL (released()), this, SLOT (handleButton()));
    m_button->setText("Example");
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::handleButton() {
    m_button->setText("Example");
    m_button->resize(100,100);

}
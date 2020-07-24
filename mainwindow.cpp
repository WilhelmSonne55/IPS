#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QCoreApplication>

#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QDebug>
#include <QToolBar>
#include <QImage>
#include <QPixmap>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QSaveFile>
#include <QDockWidget>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    //event
    action();

    //menu
    menu();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::action()
{
    //action
    QAction* Open = new QAction(QIcon(":/images/doc-open"), tr("&Open..."), this);
    QAction* Save = new QAction(QIcon(":/images/doc-open"), tr("&Save..."), this);

    //action tips
    Open->setShortcuts(QKeySequence::Open);
    Open->setStatusTip(tr("Open an Image"));

    Save->setStatusTip(tr("Save an Image"));

    //event
    connect(Open, &QAction::triggered, this, &MainWindow::open);
    connect(Save, &QAction::triggered, this, &MainWindow::save);

    //1st menu
    QMenu *OperationMenu = menuBar()->addMenu(tr("&File"));
    OperationMenu->addAction(Open);
    OperationMenu->addAction(Save);

    //2nd menu
    QMenu *ColorMenu = menuBar()->addMenu(tr("&Color"));
    ColorMenu->addAction(Open);
    ColorMenu->addAction(Save);

}

void MainWindow::menu()
{
    //status bar
    QStatusBar *pBar = statusBar();

    //label
    MainLabel = new QLabel(this);
    MainLabel->setText("Coded by William Sun");    
    pBar->addWidget(MainLabel);

    //right processing menu
    QDockWidget *pdock = new QDockWidget(this);
    addDockWidget(Qt::RightDockWidgetArea, pdock);
    pdock->setFeatures(QDockWidget::DockWidgetVerticalTitleBar|
    QDockWidget::DockWidgetFloatable);
    pdock->setFloating(true);
    pdock->setGeometry(WINDOW_X + WINDOW_WIDTH + 150, 
    WINDOW_Y, 150,WINDOW_HEIGHT);
}

bool MainWindow::maybeSave()
{
    if (!textEdit->document()->isModified())
        return true;
    const QMessageBox::StandardButton ret
        = QMessageBox::warning(this, tr("Application"),
                               tr("The document has been modified.\n"
                                  "Do you want to save your changes?"),
                               QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    switch (ret) {
    case QMessageBox::Save:
        return save();
    case QMessageBox::Cancel:
        return false;
    default:
        break;
    }
    return true;
}

void MainWindow::open(){
        QString fileName = QFileDialog::getOpenFileName(this);
        MainImage = new QImage (fileName);
        QPixmap mp;
        mp = mp.fromImage(*MainImage);
        MainLabel->setPixmap(mp.scaled(200,200,Qt::KeepAspectRatio));
        MainLabel->show();
}

void MainWindow::loadFile(const QString &fileName)
{
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        QMessageBox::warning(this, tr("Application"),
                             tr("Cannot read file %1:\n%2.")
                             .arg(QDir::toNativeSeparators(fileName), file.errorString()));
        return;
    }

    QTextStream in(&file);
#ifndef QT_NO_CURSOR
    QGuiApplication::setOverrideCursor(Qt::WaitCursor);
#endif
    textEdit->setPlainText(in.readAll());
#ifndef QT_NO_CURSOR
    QGuiApplication::restoreOverrideCursor();
#endif

    setCurrentFile(fileName);
    statusBar()->showMessage(tr("File loaded"), 2000);
}

bool MainWindow::save()
{
    if (curFile.isEmpty()) {
        return saveAs();
    } else {
        return saveFile(curFile);
    }
}

void MainWindow::setCurrentFile(const QString &fileName)
{
    curFile = fileName;
    textEdit->document()->setModified(false);
    setWindowModified(false);

    QString shownName = curFile;
    if (curFile.isEmpty())
        shownName = "untitled.txt";
    setWindowFilePath(shownName);
}

bool MainWindow::saveAs()
{
    QFileDialog dialog(this);
    dialog.setWindowModality(Qt::WindowModal);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    if (dialog.exec() != QDialog::Accepted)
        return false;
    return saveFile(dialog.selectedFiles().first());
}

bool MainWindow::saveFile(const QString &fileName)
{
    QString errorMessage;

    QGuiApplication::setOverrideCursor(Qt::WaitCursor);
    QSaveFile file(fileName);
    if (file.open(QFile::WriteOnly | QFile::Text)) {
        QTextStream out(&file);
        out << textEdit->toPlainText();
        if (!file.commit()) {
            errorMessage = tr("Cannot write file %1:\n%2.")
                           .arg(QDir::toNativeSeparators(fileName), file.errorString());
        }
    } else {
        errorMessage = tr("Cannot open file %1 for writing:\n%2.")
                       .arg(QDir::toNativeSeparators(fileName), file.errorString());
    }
    QGuiApplication::restoreOverrideCursor();

    if (!errorMessage.isEmpty()) {
        QMessageBox::warning(this, tr("Application"), errorMessage);
        return false;
    }

    setCurrentFile(fileName);
    statusBar()->showMessage(tr("File saved"), 2000);
    return true;
}
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QCoreApplication>

#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QDebug>
#include <QToolBar>
#include <QImage>
#include <QColor>
#include <QPixmap>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QSaveFile>
#include <QDockWidget>
#include <QLCDNumber>

//========================================
// Define
//========================================
#define BOUND(x, min, max) (x < min? min: (x > max? max: x))

//========================================
// Local Parameters
//========================================
#define PICTURE_WIDTH 910
#define PICTURE_HEIGHT 540

static int _HueSlideValue = 0;
static int _SatSlideValue = 0;
static int _BriSlideValue = 0;
//========================================
//  Function
//========================================
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //menu
    QStatusBar *pBar = statusBar();
    MainLabel = new QLabel(this);
    MainLabel->setText("Coded by William Sun");    
    pBar->addWidget(MainLabel);

    //event
    action();
    slidemenu();
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

    //1st menu
    QMenu *OperationMenu = menuBar()->addMenu(tr("&File"));
    OperationMenu->addAction(Open);
    OperationMenu->addAction(Save);

    //event
    connect(Open, &QAction::triggered, this, &MainWindow::open);
    connect(Save, &QAction::triggered, this, &MainWindow::save);
}

void MainWindow::slidemenu()
{
    //right processing menu
    QWidget *SlideBar = new QWidget;
    SlideBar->setWindowTitle("HSY Adjustment");
    SlideBar->setGeometry(WINDOW_X+WINDOW_WIDTH + 100, WINDOW_Y + 100, \
    500, 140);
    SlideBar->setMaximumSize(500,140);
    SlideBar->setMinimumSize(500,140);

    QSlider *HueSlider = new QSlider(Qt::Horizontal, SlideBar);
    HueSlider->setRange(-180,180);
    HueSlider->setValue(0);
    HueSlider->setGeometry(120,30,300,20);

    QSlider *BriSlider = new QSlider(Qt::Horizontal, SlideBar);
    BriSlider->setRange(-50,50);
    BriSlider->setValue(0);
    BriSlider->setGeometry(120,60,300,20);

    QSlider *SatSlider = new QSlider(Qt::Horizontal, SlideBar);
    SatSlider->setRange(-50,50);
    SatSlider->setValue(_SatSlideValue);
    SatSlider->setGeometry(120,90,300,20);

    //event
    if(MainImage)
    {
        connect(HueSlider, SIGNAL(valueChanged(int)), this, SLOT(hue(int)));
        connect(SatSlider, SIGNAL(valueChanged(int)), this, SLOT(saturation(int)));
        connect(BriSlider, SIGNAL(valueChanged(int)), this, SLOT(brightness(int)));
    }

    QLCDNumber *lcd = new QLCDNumber(this);
    lcd->setGeometry(70, 20, 100, 30);
    QObject::connect(SatSlider, SIGNAL(valueChanged(int)), lcd, SLOT(display(int)));

    SlideBar->show();
}

void MainWindow::hue(int delta)
{
    QImage* newImage = new QImage(MainImage->width(), MainImage->height(), QImage::Format_RGB888);

    for(int x = 0; x < newImage->width(); x++)
    {
        for(int y = 0; y < newImage->height(); y++)
        {
            QColor oldColor = QColor(MainImage->pixel(x,y));
            QColor newColor = oldColor.toHsl();
            int H, S, L;

            H = BOUND(newColor.hue() + delta, 0, 360);
            S = BOUND(newColor.saturation() + _SatSlideValue, 0, 255);
            L = BOUND(newColor.lightness() + _BriSlideValue, 0, 255);
            newColor.setHsl(H,S,L);

            newImage->setPixel(x, y, qRgb(newColor.red(), newColor.green(), newColor.blue()));
        }
    }

    _HueSlideValue = delta;

    QPixmap mp;
    mp = mp.fromImage(*newImage);
    MainLabel->setPixmap(mp.scaled(PICTURE_WIDTH,PICTURE_HEIGHT,Qt::KeepAspectRatio));
    MainLabel->show();
    delete newImage;
}

void MainWindow::brightness(int delta)
{
    QImage* newImage = new QImage(MainImage->width(), MainImage->height(), QImage::Format_RGB888);

    for(int x = 0; x < newImage->width(); x++)
    {
        for(int y = 0; y < newImage->height(); y++)
        {
            QColor oldColor = QColor(MainImage->pixel(x,y));
            QColor newColor = oldColor.toHsl();
            int H, S, L;

            H = newColor.hue();
            S = BOUND(newColor.saturation() + _SatSlideValue, 0, 255);
            L = BOUND(newColor.lightness() +delta, 0, 255);
            newColor.setHsl(H,S,L);

            newImage->setPixel(x, y, qRgb(newColor.red(), newColor.green(), newColor.blue()));
        }
    }

    _BriSlideValue = delta;

    QPixmap mp;
    mp = mp.fromImage(*newImage);
    MainLabel->setPixmap(mp.scaled(PICTURE_WIDTH,PICTURE_HEIGHT,Qt::KeepAspectRatio));
    MainLabel->show();
    delete newImage;
}

void MainWindow::saturation(int delta)
{
    QImage* newImage = new QImage(MainImage->width(), MainImage->height(), QImage::Format_RGB888);

    for(int x = 0; x < newImage->width(); x++)
    {
        for(int y = 0; y < newImage->height(); y++)
        {
            QColor oldColor = QColor(MainImage->pixel(x,y));
            QColor newColor = oldColor.toHsl();
            int H, S, L;

            H = newColor.hue();
            S = BOUND(newColor.saturation()+ delta, 0, 255);
            L = BOUND(newColor.lightness() + _BriSlideValue, 0, 255);
            newColor.setHsl(H,S,L);

            newImage->setPixel(x, y, qRgb(newColor.red(), newColor.green(), newColor.blue()));
        }
    }
    qDebug()<<"delta"<<delta<<endl;

    _SatSlideValue = delta;

    QPixmap mp;
    mp = mp.fromImage(*newImage);
    MainLabel->setPixmap(mp.scaled(PICTURE_WIDTH,PICTURE_HEIGHT,Qt::KeepAspectRatio));
    MainLabel->show();
    delete newImage;
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
        MainLabel->setPixmap(mp.scaled(PICTURE_WIDTH,PICTURE_HEIGHT,Qt::KeepAspectRatio));
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
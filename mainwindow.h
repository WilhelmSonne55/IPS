#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QString>
#include <QLabel>

#define WINDOW_HEIGHT 500
#define WINDOW_WIDTH  500
#define WINDOW_X  500
#define WINDOW_Y  100

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    void loadFile(const QString &fileName);

    ~MainWindow();
private:


private slots:
    void action();
    void menu();

    bool maybeSave();
    void open();
    bool save();
    bool saveAs();
    bool saveFile(const QString &fileName);

private:
    QImage *MainImage;
    Ui::MainWindow *ui;
    QPushButton *m_button;
    QPlainTextEdit *textEdit;
    QString curFile;
    QLabel *MainLabel;

    void setCurrentFile(const QString &fileName);
};

#endif // MAINWINDOW_H

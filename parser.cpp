#include "parser.h"
#include<iostream>
using namespace std;

Parser::Parser()
{
    cout<<"parser construct"<<endl;
}

void Parser::demosaic()
{
    cout<<"base demosaic"<<endl;
}

ParserImpl::ParserImpl()
{
    cout<<"ParserImpl construct"<<endl;

}

void ParserImpl::demosaic()
{
    cout<<"derived demosaic"<<endl;
}